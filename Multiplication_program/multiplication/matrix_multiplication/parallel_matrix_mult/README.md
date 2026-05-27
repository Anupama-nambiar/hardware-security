<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Parallel Dual Program Design

This directory contains a design that runs **two independent matrix multiplications concurrently** on a single Ryzen AI NPU (Phoenix/Hawk). The NPU's 4×4 AI Engine array is partitioned into two equal halves — each half executes a fully independent matrix multiplication with its own tiles, ObjectFIFOs, and DMA buffer descriptors. Both programs' DMA transfers are issued before any synchronization barrier, so they execute in true parallel.

```
NPU 4×4 AI Engine Array
┌────────────────────────────-----------┐
│  Col 0     Col 1  │  Col 2     Col 3  │
│  ─────────────────┼───────────────────│
│    Program 0      │     Program 1     │
│   (shim 0–1)      │   (shim 2–3)      │
│   (mem  0–1)      │   (mem  2–3)      │
│   (cores x8)      │   (cores x8)      │
└──────────────────────────-----------──┘
```

Both programs operate on **identically sized** matrices, configured via the same `-M`, `-K`, `-N`, `-m`, `-k`, `-n` Makefile arguments. The two programs' input and output data are packed into **combined host buffers** (`A_combined`, `B_combined`, `C_combined`) to avoid a hardware limitation where the compiler can merge same-typed, same-sized buffer objects, causing DMA deadlocks.

---

## Building and Running the Design

With the default configuration, both programs perform matrix-matrix multiplication on `int16` inputs (`int16` output) with a `64`×`64` default tiling size.

You will need C++23 for `bfloat16_t` support in `test.cpp`, available in `g++-13`: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu)

```shell
make
make whole_array_parallel.exe
make run
```


---

## Detailed Design Explanation

The configuration of the AI Engine array is described in [`whole_array.py`](./whole_array.py). The test harness is in [`test.cpp`](./test.cpp).

> **Note:** The term "tile" has two distinct meanings below:
> - *AIE tiles* are hardware components (Shim, Memory, Compute tiles).
> - *Matrix tiles* are sub-matrices produced by tiling the input/output matrices.

---

### 1. Array Partitioning and Program Structure

The design defines three top-level constants that govern how the array is split:

```python
N_AIE_ROWS    = 4  # fixed for NPU — 4 compute rows per column
COLS_PER_PROG = 2  # each program occupies 2 columns
N_PROGRAMS    = 2  # two programs run in parallel
```

**Program 0** occupies columns 0–1. **Program 1** occupies columns 2–3. Each program is fully self-contained: it has its own shim tiles, memory tiles, compute tiles, ObjectFIFOs, and DMA buffer descriptors. They share no hardware resources and require no synchronization with each other during computation.

Each program uses `COLS_PER_PROG × N_AIE_ROWS = 8` active compute cores.

---

### 2. Defining Matrix Dimensions and Data Types

Both programs receive the same matrix dimensions, passed from the Makefile:

| Matrix       | Size        | Core Submatrix Size | Vector Intrinsic Size |
|--------------|-------------|---------------------|-----------------------|
| `A` (Input)  | `M` × `K`   | `m` × `k`           | `r` × `s`             |
| `B` (Input)  | `K` × `N`   | `k` × `n`           | `s` × `t`             |
| `C` (Output) | `M` × `N`   | `m` × `n`           | `r` × `t`             |

The vector intrinsic dimensions `(r, s, t)` are fixed by the hardware and data type:

| `dtype_in` | `r` | `s` | `t` |
|------------|-----|-----|-----|
| `i16`      | 4   | 4   | 4   |
| `i8`       | 4   | 8   | 8   |
| `bf16`     | 4   | 8   | 4   |

Tiling occurs at two levels, both driven by the Makefile parameters:

1. **Core Submatrix Chunks (`m`×`k`, `k`×`n`, `m`×`n`):** The full input matrices stream into each program's compute cores in these chunk sizes. The `-m`, `-k`, `-n` values passed from the Makefile directly determine the sizes of the ObjectFIFOs, the DMA transfer dimensions, and the inner loop trip count (`K // k`) inside each core. Changing these at build time reshapes the entire data movement pipeline for both programs simultaneously.

2. **Vector Intrinsic Size (`r`×`s`, `s`×`t`):** The memory tiles perform an in-flight re-layout of each `m`×`k` chunk into contiguous `r`×`s` blocks as data passes from L2 to L1 memory. This transformation is expressed as wrap/stride pairs in the ObjectFIFO definition and has zero runtime cost.

Tiling constraints (per program):
```
M % (m * N_AIE_ROWS)    == 0   →  M % (m * 4) == 0
K % k                   == 0
N % (n * COLS_PER_PROG) == 0   →  N % (n * 2) == 0
m % r == 0,  k % s == 0,  n % t == 0
```

---

### 3. Combined Host Buffers

A critical implementation detail is that both programs' data is packed into **three combined host buffers** rather than six separate ones:

```
A_combined  =  [ A0  (M*K elements) | A1  (M*K elements) ]
B_combined  =  [ B0  (K*N elements) | B1  (K*N elements) ]
C_combined  =  [ C0  (M*N elements) | C1  (M*N elements) ]
```

This is necessary because the NPU shim DMA hardware supports a maximum of 16 buffer descriptors (BDs) per channel, which the ping/pong transfer scheme splits into two groups of 8 (BD IDs 0–7 per phase). With 6 separate buffers (A0, B0, C0, A1, B1, C1), the runtime sequence would attempt to assign BD IDs beyond index 7 within a single ping/pong phase, exceeding this hardware limit. This produces a hard crash at runtime. 

By consolidating to 3 combined buffers, the number of BD assignments per phase stays within the hardware limit of 8. Each program accesses its slice via an offset computed at sequence time, with no additional BD cost.

```python
A_offsets = [0,        A0_elems]   # Program 0 starts at 0, Program 1 after A0
B_offsets = [0,        B0_elems]
C_offsets = [0,        C0_elems]
```

---

### 4. Defining Data Movement Inside the NPU

For each program `prog_idx` (0 or 1), the column base is `col_base = prog_idx * COLS_PER_PROG`. All tiles and ObjectFIFOs are indexed relative to this base, so the two programs' data paths are entirely disjoint.

The data path for each program follows the same three-level hierarchy:

```
Host (external DRAM) — combined buffer, offset by program
        │
        ▼  [npu_dma_memcpy_nd]
Shim Tiles (row 0, cols col_base .. col_base+1)
        │
        ▼  A_L3L2 / B_L3L2 ObjectFIFOs
Memory Tiles (row 1)  ◄── layout transform (row-major → r×s / s×t blocks)
        │
        ▼  A_L2L1 / B_L2L1 ObjectFIFOs
Compute Tiles (rows 2–5, cols col_base .. col_base+1)
        │
        ▼  C_L1L2 ObjectFIFOs
Memory Tiles (row 1)  ◄── layout transform (r×t blocks → row-major)
        │
        ▼  C_L2L3 ObjectFIFOs
Shim Tiles (row 0) → Host
```

The following ObjectFIFOs are created **per program** (named with a `p{prog_idx}` prefix to prevent name collisions):

- **`A_L3L2_p{i}`** (one per column in the program): Moves `m * k * n_A_tiles_per_shim`-sized tiles of `A` from the shim tile to the memory tile. Since `COLS_PER_PROG = 2` and `N_AIE_ROWS = 4`, each shim handles `n_A_tiles_per_shim = 2` row-tiles of `A`.

- **`A_L2L1_p{i}`** (one per AIE row): Broadcasts an `m`×`k` tile of `A` from the memory tile to both compute cores in the same row (across the 2 active columns), after transforming from row-major to `r`×`s`-blocked layout. The `object_fifo_link()` from `A_L3L2` distributes the two row-tiles from each shim to the correct pair of `A_L2L1` FIFOs using offsets `[0, m*k]`.

- **`B_L3L2_p{i}`** (one per column): Moves `k`×`n`-sized tiles of `B` from the shim tile to the memory tile.

- **`B_L2L1_p{i}`** (one per column): Broadcasts `k`×`n` tiles of `B` down all 4 compute rows in the same column, after transforming to `s`×`t`-blocked layout.

- **`C_L1L2_p{i}_{col}_{row}`** (one per compute core): Collects `m`×`n` result tiles from each core and delivers them to the column's memory tile. The `object_fifo_link()` from `C_L1L2` joins the 4 per-row tiles into a single `m*n*N_AIE_ROWS`-sized block using row offsets `[0, m*n, 2*m*n, 3*m*n]`.

- **`C_L2L3_p{i}`** (one per column): Moves the assembled output block from the memory tile back to the shim tile and out to the host, transforming from `r`×`t` blocks back into row-major layout.

#### Data Layout Transformations

For matrix `A`, the `A_L2L1` FIFO re-lays out each `m`×`k` row-major tile into contiguous `r`×`s` blocks via these wrap/stride pairs:

```python
[
    (m // r, r * k),   # outer: step between block-rows along m
    (k // s, s),       # step between blocks along k
    (r, k),            # rows within one r×s block
    (s, 1),            # contiguous elements within a row
]
```

For matrix `B`, the `B_L2L1` FIFO applies the equivalent transformation with `s`×`t` blocks (or a transposed variant if `--b-col-maj` is set). For matrix `C`, the `C_L2L3` FIFO reverses the process, assembling `r`×`t` blocks back into row-major output.

---

### 5. Defining Core Computations

Each of the 8 active compute cores per program runs the same `core_body()` function. A `make_core_body()` factory function is used to correctly capture the per-core loop variables (avoiding Python closure capture issues in a loop):

1. Acquire an output buffer slot from `C_L1L2` → `elem_out`.
2. Zero out `elem_out` to clear stale values.
3. For each of the `K // k` steps along the shared `K` dimension:
   - Acquire the next `m`×`k` tile of `A` from `A_L2L1[row]` → `elem_in_a`.
   - Acquire the next `k`×`n` tile of `B` from `B_L2L1[col]` → `elem_in_b`.
   - Call the microkernel: `matmul(elem_in_a, elem_in_b, elem_out)` — accumulates into `elem_out`.
   - Release `elem_in_a` and `elem_in_b`.
4. Release `elem_out` to `C_L1L2`, making the result available to the memory tile.

This loop runs indefinitely (`range_(0xFFFFFFFF)`), processing `n_tiles_per_core` output tiles per outer iteration, where:
```
n_tiles_per_core = (M // m) * (N // n) // (N_AIE_ROWS * COLS_PER_PROG)
```

---

### 6. Defining External Data Transfer Sequences

The `runtime_sequence` function receives the three combined host buffers and issues DMA transfers for **both programs before any `dma_wait`**, so both programs' data movement begins concurrently.

Transfers are organized into transfer blocks of at most `tb_max_n_rows = 4` tile rows to stay within the hardware's buffer descriptor (BD) limit. Within each block, a **ping/pong** scheme double-buffers the `A` transfers: while one half of the BDs actively move data, the other half are reconfigured for the next block, overlapping reconfiguration with data movement.

For each transfer block `tb` and each ping/pong phase, the sequence iterates over **both programs** and **both columns** within each program, issuing all BDs before any synchronization:

- **Matrix C (output):** Each shim tile collects an `(m * N_AIE_ROWS)`-×-`n` output block from `C_L2L3` and writes it to the correct location in `C_combined`, offset by the program's `C_base`. Sizes and strides tile the full `M`×`N` result, stepping by `n * COLS_PER_PROG` across columns and `m * N_AIE_ROWS * N` across row blocks.

- **Matrix A (input):** Each shim tile reads an `(m * n_A_tiles_per_shim)`-row block from `A_combined`, offset by both the program's `A_base` and the tile row within the block. The outermost DMA dimension has stride `0`, encoding a broadcast of the same rows of `A` across all `N // n // COLS_PER_PROG` passes through the `N` dimension — `A` is reused without re-reading from host memory.

- **Matrix B (input):** Each shim tile reads `n`-wide column slices from `B_combined`, offset by the program's `B_base`. Adjacent columns within a program take interleaved non-overlapping slices, stepping by `n * COLS_PER_PROG` across `N`.

After all BDs for both programs are issued for a given transfer block, a single `dma_wait` synchronizes on all `C_L2L3` FIFOs from both programs together:

```python
dma_wait(*all_C_l2l3_fifos[0], *all_C_l2l3_fifos[1])
```

This ensures that both programs complete their output transfers before the next block's BDs are reused.

---

### 7. Benchmarking with Multiple Input Types

The `test.cpp` harness measures **wall-clock execution time** for both programs running in parallel, using `std::chrono`. Three structured input types are tested:

| Input Type   | Matrix `A`                          | Matrix `B`                          | Expected `C`       |
|--------------|-------------------------------------|-------------------------------------|--------------------|
| **Zeros**    | All elements = 0                    | All elements = 0                    | All zeros          |
| **Diagonal** | Identity-like (1s on diagonal)      | Identity-like (1s on diagonal)      | Identity-like      |
| **Ones**     | All elements = 1                    | All elements = 1                    | Each element = `K` |

Both programs receive the same input type in each test run. Results for each program are verified independently against the expected output. The wall-clock timer covers the full submission-to-completion window for both programs together, so the reported time reflects true concurrent execution.

---

## Key Parameters (from Makefile)

| Parameter      | Default | Description                                               |
|----------------|---------|-----------------------------------------------------------|
| `-M`           | 512     | Rows of matrix `A` / `C` (per program)                    |
| `-K`           | 512     | Shared inner dimension of `A` and `B` (per program)       |
| `-N`           | 512     | Columns of matrix `B` / `C` (per program)                 |
| `-m`           | 64      | Rows per core sub-tile of `A` / `C`                       |
| `-k`           | 64      | Columns per core sub-tile of `A` / rows of `B`            |
| `-n`           | 32      | Columns per core sub-tile of `B` / `C`                    |
| `--dtype_in`   | `i16`   | Input data type (`i8`, `i16`, `bf16`)                     |
| `--dtype_out`  | `i16`   | Output data type (`i16`, `i32`, `f32`, `bf16`)            |
| `--b-col-maj`  | 0       | Use column-major layout for `B` (0 = row-major)           |
| `--c-col-maj`  | 0       | Use column-major layout for `C` output (0 = row-major)    |

Both programs always use `COLS_PER_PROG = 2` columns and `N_AIE_ROWS = 4` rows, consuming the full 4×4 NPU array. Program A uses columns 0-1 and Program B uses columns 2-3
