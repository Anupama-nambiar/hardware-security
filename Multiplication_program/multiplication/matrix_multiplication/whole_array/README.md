<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Configurable Column Design

This directory contains a matrix multiplication design for a Ryzen AI device with an NPU (Neural Processing Unit). Unlike the whole-array design, this variant allows precise control over **which columns of AI Engines (AIEs) are used** via a configurable column offset and column count. This makes it suitable for partitioned workloads, resource isolation experiments, and performance benchmarking across different AIE column configurations.

At a high level, the code does the following (in order):

1. [**Defining Matrix Dimensions and Data Types:**](#1-defining-matrix-dimensions-and-data-types) We specify the dimensions `M`, `K`, `N` for input matrices `A` (`M`×`K`) and `B` (`K`×`N`), and output matrix `C` (`M`×`N`), along with their data type. Large matrices are split into sub-matrix tiles at two levels: `m`×`k`, `k`×`n`, and `m`×`n` for per-core chunks, and `r`×`s`, `s`×`t` for the vector intrinsic size required by the AIE hardware.

2. [**Constructing a Configurable AIE Array:**](#2-constructing-a-configurable-aie-array) Based on the `--col_offset` and `--n-aie-cols` parameters, the design selects a specific contiguous slice of columns from the NPU's 4×4 (or 4×8 for NPU2) array. Shim tiles, memory tiles, and compute tiles are all allocated within this column range.

3. [**Defining Data Movement Inside the NPU:**](#3-defining-data-movement-inside-the-npu) ObjectFIFOs move data between shim tiles, memory tiles, and compute tiles for matrices `A`, `B`, and `C`. Data flows in as `m`×`k` and `k`×`n`-sized chunks and flows out as `m`×`n`-sized results.

4. [**Defining Core Computations:**](#4-defining-core-computations) Each active compute core runs an inner loop that acquires input tiles from its ObjectFIFOs, calls the matrix multiply microkernel, accumulates results, and releases the output tile.

5. [**Defining External Data Transfer Sequences:**](#5-defining-external-data-transfer-sequences) The `aie.runtime_sequence()` op configures DMA transfers from host memory into the AIE array and back, tiling the full matrices into the `m`×`k` and `k`×`n` sub-matrix chunks required by each column's compute cores.

6. [**Benchmarking with Multiple Input Types:**](#6-benchmarking-with-multiple-input-types) The `test.cpp` file drives the design and measures **wall-clock execution time** across three structured input types — zeros, diagonal (identity-like), and ones — to characterize performance under different data conditions.

---

## Building and Running the Design

With the default configuration, this design performs matrix-matrix multiplication on `int16` inputs (`int32` output). The tiling size is `64`×`64` by default, using 2 AIE columns starting at column offset 0.

You will need C++23 for `bfloat16_t` support in `test.cpp`, available in `g++-13`: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu)

To compile and run:

```shell
make
make whole_array.exe
make run
```

To run with a custom column configuration (e.g., 2 columns starting at column 2):

```shell
make run COL_OFFSET=2 N_AIE_COLS=2
```

---

## Detailed Design Explanation

The configuration of the AI Engine array is described in [`whole_array.py`](./whole_array.py). The test harness is in [`test.cpp`](./test.cpp).

> **Note:** The term "tile" has two distinct meanings below:
> - *AIE tiles* are hardware components (Shim, Memory, Compute tiles).
> - *Matrix tiles* are sub-matrices produced by tiling the input/output matrices.

---

### 1. Defining Matrix Dimensions and Data Types

The following constants define matrix and tiling sizes:

| Matrix       | Size        | Core Submatrix Size | Vector Intrinsic Size |
|--------------|-------------|---------------------|-----------------------|
| `A` (Input)  | `M` × `K`   | `m` × `k`           | `r` × `s`             |
| `B` (Input)  | `K` × `N`   | `k` × `n`           | `s` × `t`             |
| `C` (Output) | `M` × `N`   | `m` × `n`           | `r` × `t`             |

Tiling occurs at two levels:

1. **Core Submatrix Chunks (`m`×`k`, `k`×`n`, `m`×`n`):** The input matrices are split into sub-tiles that stream into each AIE compute core. Each core works on its assigned sub-tile concurrently, and results are reassembled into the full output matrix `C`. The sizes of these tiles are passed in from the Makefile (via `-m`, `-k`, `-n` arguments) and directly control how the DMA transfers and ObjectFIFOs are configured. Changing these values reshapes the entire data movement pipeline at compile time.

2. **Vector Intrinsic Size (`r`×`s`, `s`×`t`):** Within each core, the computation is performed using hardware MAC (multiply-accumulate) vector instructions. These require data to be laid out in `r`×`s` and `s`×`t`-sized blocks. The memory tiles perform an in-flight data layout transformation as data passes from L2 to L1 memory, so that this re-layout has zero runtime cost.

---

### 2. Constructing a Configurable AIE Array

This design introduces two key parameters:

- **`--col_offset`**: The index of the first AIE column to use (0-indexed). For example, `--col_offset 2` starts tile allocation at column 2.
- **`--n-aie-cols`**: The number of AIE columns to use (1, 2, or 4 for NPU; up to 8 for NPU2).

These parameters are passed from the Makefile and propagate through the design, affecting all tile declarations, ObjectFIFO endpoints, and DMA buffer descriptor assignments.

The NPU hardware is structured as a 6-row × 4-column array (or 6×8 for NPU2):

- **Shim tiles (row 0):** Interface with the host for all DMA data movement. Only the shim tiles within the selected column range are activated.
- **Memory tiles (row 1):** Stage and redistribute data between the host-facing shim tiles and the compute cores. They also perform the data layout transformation from row-major to vector-intrinsic format.
- **Compute tiles (rows 2–5):** Each column contains 4 rows of compute cores. With `n_aie_cols` columns selected, this gives `n_aie_cols × 4` total active cores.

The device type (`npu1_1col`, `npu1_2col`, `npu1`, or `npu2`) is automatically selected based on the sum `col_offset + n_aie_cols`.

---

### 3. Defining Data Movement Inside the NPU

Data movement is described using **ObjectFIFOs**, a hardware-level abstraction that manages DMA configuration, buffer allocation, and lock synchronization. ObjectFIFOs behave as First-In-First-Out queues between AIE components.

The data path for each matrix is:

```
Host (external DRAM)
      │
      ▼  [DMA via npu_dma_memcpy_nd]
Shim Tiles (row 0)
      │
      ▼  A_L3L2 / B_L3L2 / C_L2L3 ObjectFIFOs
Memory Tiles (row 1)   ◄── layout transform (row-major → r×s blocks)
      │
      ▼  A_L2L1 / B_L2L1 / C_L1L2 ObjectFIFOs
Compute Tiles (rows 2–5)
```

Specifically, the following ObjectFIFOs are created per active column:

- **`A_L3L2`**: Moves `m`×`k`-sized tiles of matrix `A` from the shim tile to the memory tile.
- **`A_L2L1`**: Broadcasts `m`×`k` tiles of `A` from the memory tile to all compute cores in the same row, after applying the layout transformation to `r`×`s` blocks.
- **`B_L3L2`**: Moves `k`×`n`-sized tiles of matrix `B` from the shim tile to the memory tile.
- **`B_L2L1`**: Broadcasts `k`×`n` tiles of `B` from the memory tile to all compute cores in the same column, after applying the layout transformation to `s`×`t` blocks.
- **`C_L1L2`**: Collects `m`×`n` output tiles from each compute core and sends them to the memory tile.
- **`C_L2L3`**: Combines the per-core output tiles in the memory tile and moves the assembled result back to the shim tile and out to the host, transforming from `r`×`t` blocks back to row-major layout.

The `object_fifo_link()` operation connects the `L3L2` and `L2L1` FIFOs (and `L1L2` to `L2L3` for `C`) so that data flows through the memory tile automatically. This is what enables the zero-cost data layout transformation: the memory tile DMA performs the re-layout as part of the transfer, with no separate copy step.

#### Tiling and Data Layout Transformations

Input data is stored in **row-major format** in host memory. Before computation, matrix tiles must be re-laid out so that the small `r`×`s` (and `s`×`t`) blocks required by the vector MAC instructions are contiguous in AIE local memory. This transformation is expressed as a sequence of `(wrap, stride)` pairs in the ObjectFIFO definition.

For matrix `A`, the transformation from a `m`×`k` row-major tile into `r`×`s`-sized blocks is:

```python
[
    (m // r, r * k),   # Step over rows of r×s blocks
    (k // s, s),       # Step over columns of r×s blocks
    (r, k),            # Transfer one row of an r×s block
    (s, 1),            # Transfer contiguous elements within that row
]
```

Reading back-to-front: the innermost dimension transfers `s` contiguous elements (one row of an `r`×`s` block); the next steps down `r` rows to assemble the full block; the next advances to the next block along the `k` dimension; and the outermost steps down to the next block-row along `m`. The result is that each `r`×`s` sub-block arrives in the compute core's local memory as a contiguous region.

Matrix `B` undergoes an equivalent transformation with dimensions `k`×`n` and block size `s`×`t`. If `--b-col-maj` is set, a transposed variant is used instead. Output matrix `C` is transformed back from `r`×`t` blocks into row-major upon egress through the memory tile.

---

### 4. Defining Core Computations

Each active compute core runs the same `core_body()` function, which loops indefinitely and processes one `m`×`n` output tile per iteration:

1. Acquire an output buffer slot from `C_L1L2` (`elem_out`).
2. Zero out `elem_out` to clear any stale values.
3. For each of the `K // k` steps along the shared `K` dimension:
   - Acquire the next `m`×`k` tile of `A` from `A_L2L1` (`elem_in_a`).
   - Acquire the next `k`×`n` tile of `B` from `B_L2L1` (`elem_in_b`).
   - Call the matrix multiply microkernel: `matmul(elem_in_a, elem_in_b, elem_out)`. The result is accumulated in `elem_out`.
   - Release `elem_in_a` and `elem_in_b` back to their FIFOs.
4. Release `elem_out` to the `C_L1L2` FIFO, making the result available to the memory tile.

---

### 5. Defining External Data Transfer Sequences

The `aie.runtime_sequence()` op configures the host-side DMA transfers that feed data into and collect results from the AIE array. The `npu_dma_memcpy_nd` function is used throughout, specifying multi-dimensional offsets, sizes, and strides to tile the full `M`×`K` and `K`×`N` host matrices into `m`×`k` and `k`×`n` sub-tiles, and to reassemble the `m`×`n` output tiles into the full `M`×`N` result matrix.

To stay within the hardware limit on buffer descriptors (BDs), transfers are grouped into blocks of at most 4 tile rows (`tb_max_n_rows`). Within each block, transfers are further divided into **ping** and **pong** phases. While one half of the BDs are actively transferring data, the other half can be reconfigured for the next block. This double-buffering overlap is especially beneficial for large matrices.

For each active AIE column `col`:

- **Matrix A:** Each shim tile transfers a block of `(m × n_A_tiles_per_shim)`-row tiles from the host. The transfer repeats `N // n // n_aie_cols` times along the `N` dimension (since the same rows of `A` are reused for every group of `B` columns). Sizes and strides are: `[N//n//n_aie_cols, K//k, m*n_A_tiles_per_shim, k]` with strides `[0, k, K, 1]` — the zero stride on the outermost dimension encodes the broadcast of `A` across all `N`-dimension passes without re-reading from host memory.

- **Matrix B:** Each shim tile transfers `n`-wide column blocks from the host, stepping by `n * n_aie_cols` across `N` so that adjacent columns take interleaved non-overlapping slices. Sizes and strides are: `[N//n//n_aie_cols, K//k, k, n]` with strides `[n*n_aie_cols, k*N, N, 1]`.

- **Matrix C:** Each shim tile collects an `(m*n_aie_rows)`-×-`n`-sized output block and writes it back to the correct location in the host `M`×`N` buffer, stepping by `n * n_aie_cols` across columns and `m * n_aie_rows * N` across rows.

After all columns' BDs are issued for a given transfer block, `dma_wait` synchronizes on all `C_L2L3` FIFOs before the next block begins.

---

### 6. Benchmarking with Multiple Input Types

The `test.cpp` harness measures **wall-clock execution time** using C++ `std::chrono`. Three structured input types are tested to characterize how data content affects end-to-end latency:

| Input Type   | Matrix `A`           | Matrix `B`           | Expected `C`              |
|--------------|----------------------|----------------------|---------------------------|
| **Zeros**    | All elements = 0     | All elements = 0     | All zeros                 |
| **Diagonal** | Identity-like (1s on diagonal, 0s elsewhere) | Identity-like | Passthrough of the other matrix |
| **Ones**     | All elements = 1     | All elements = 1     | Each element = `K`        |

For each input type, the test initializes the host buffers, submits the computation to the NPU, and records the elapsed wall-clock time from submission to completion. Results are printed alongside a correctness check against the expected output, so both performance and accuracy can be verified in a single run.

This benchmarking approach is useful for isolating whether performance varies with data sparsity or structure, and for establishing a baseline before optimizing tiling parameters or column configurations.

---

## Key Parameters (from Makefile)

| Parameter       | Default | Description                                               |
|-----------------|---------|-----------------------------------------------------------|
| `-M`            | 512     | Number of rows in matrix `A` and `C`                      |
| `-K`            | 512     | Shared inner dimension of `A` and `B`                     |
| `-N`            | 512     | Number of columns in matrix `B` and `C`                   |
| `-m`            | 64      | Rows per core sub-tile of `A` / `C`                       |
| `-k`            | 64      | Columns per core sub-tile of `A` / rows of `B`            |
| `-n`            | 32      | Columns per core sub-tile of `B` / `C`                    |
| `--n-aie-cols`  | 2       | Number of AIE columns to activate                         |
| `--col_offset`  | 0       | Index of the first AIE column to use                      |
| `--dtype_in`    | `i16`   | Input data type (`i8`, `i16`, `bf16`)                     |
| `--dtype_out`   | `i16`   | Output data type (`i16`, `i32`, `f32`, `bf16`)            |

Tiling constraints that must be satisfied:

```
M % (m * 4)          == 0   # 4 AIE rows per column
K % k                == 0
N % (n * n_aie_cols) == 0
m % r == 0  and  k % s == 0  and  n % t == 0   # vector intrinsic alignment
```
