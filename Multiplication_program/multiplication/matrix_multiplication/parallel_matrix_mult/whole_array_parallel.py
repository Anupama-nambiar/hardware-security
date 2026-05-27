

# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# (c) Copyright 2025 AMD Inc.

# Runs two independent matrix multiplication programs in parallel on a single
# NPU (Phoenix/Hawk) 4x4 AI Engine array.

# The array is partitioned into two groups of 2 columns:
#   Program 0: columns 0-1  (shim rows 0-1, mem rows 0-1, core rows 0-1 x 4 rows)
#   Program 1: columns 2-3  (shim rows 2-3, mem rows 2-3, core rows 2-3 x 4 rows)

# Each program is a fully independent matmul with its own tiles, FIFOs, and
# DMA buffer descriptors. Both programs' BDs are issued before any dma_wait,
# so they execute concurrently.

# Each program uses n_aie_cols=2, n_aie_rows=4, so all original tiling rules
# still apply with those values substituted in.

# Tiling constraints per program:
#   M % (m * 4) == 0
#   K % k == 0
#   N % (n * 2) == 0


# #parallel- edited_whole_arrray mult
import argparse

import numpy as np

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

from aie.iron import str_to_dtype

# NPU microkernel MAC dimensions (r, s, t)
microkernel_mac_dim_map = {
    "bf16": (4, 8, 4),
    "i8":   (4, 8, 8),
    "i16":  (4, 4, 4),
}

N_AIE_ROWS    = 4  # fixed for NPU
COLS_PER_PROG = 2  # each program gets 2 of the 4 available columns
N_PROGRAMS    = 2  # total programs running in parallel


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Parallel Dual Matrix Multiplication (NPU npu1)",
        description=(
            "Emits MLIR for two independent matrix multiplications running in "
            "parallel on NPU's 4x4 AI Engine array, each using 2 columns."
        ),
    )
    # Core dims — passed by Makefile via -M -K -N -m -k -n
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    # Other args
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--scalar",    type=bool, default=False)
    argparser.add_argument("--dtype_in",  type=str, choices=["bf16", "i8", "i16"], default="i16")
    argparser.add_argument("--dtype_out", type=str, choices=["bf16", "i8", "i16", "f32", "i32"], default="i16")
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--generate-taps", action="store_true")
    argparser.add_argument("--dev", type=str, default="npu")
    argparser.add_argument("--n-aie-cols", type=int, default=None)
    args = argparser.parse_args()

    # Both programs get identical dims from the Makefile
    M, K, N, m, k, n = args.M, args.K, args.N, args.m, args.k, args.n

    prog_dims = [
        (M, K, N, m, k, n),
        (M, K, N, m, k, n),
    ]

    with mlir_mod_ctx() as ctx:
        maybe_taps = my_dual_matmul(
            prog_dims,
            args.dtype_in,
            args.dtype_out,
            bool(args.b_col_maj),
            bool(args.c_col_maj),
            bool(args.scalar),
            args.trace_size,
            args.generate_taps,
        )
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def my_dual_matmul(
    prog_dims,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    use_scalar,
    trace_size,
    generate_taps=False,
):
    assert len(prog_dims) == N_PROGRAMS

    dtype_in  = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), "Both dtypes must be integral or both float"
    assert np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize, \
        "Output dtype must be >= input dtype in element size"

    r, s, t = microkernel_mac_dim_map[dtype_in_str]
    fifo_depth = 1

    for p, (M, K, N, m, k, n) in enumerate(prog_dims):
        assert M % (m * N_AIE_ROWS) == 0, \
            f"Prog {p}: M={M} must be divisible by m*n_aie_rows = {m * N_AIE_ROWS}"
        assert K % k == 0, \
            f"Prog {p}: K={K} must be divisible by k={k}"
        assert N % (n * COLS_PER_PROG) == 0, \
            f"Prog {p}: N={N} must be divisible by n*cols_per_prog = {n * COLS_PER_PROG}"
        if not use_scalar:
            assert m % r == 0, f"Prog {p}: m={m} not divisible by r={r}"
            assert k % s == 0, f"Prog {p}: k={k} not divisible by s={s}"
            assert n % t == 0, f"Prog {p}: n={n} not divisible by t={t}"

    all_taps = [{"A": [], "B": [], "C": []} for _ in range(N_PROGRAMS)]

    all_A_l3l2_fifos = [None, None]
    all_B_l3l2_fifos = [None, None]
    all_C_l2l3_fifos = [None, None]

    M0, K0, N0, m0, k0, n0 = prog_dims[0]
    M1, K1, N1, m1, k1, n1 = prog_dims[1]

    # Combined buffer sizes (avoids aiecc.py merging same-typed same-sized buffers
    # into the same bo slot, which causes DMA deadlocks).
    # Layout: A_combined = [A0 | A1], B_combined = [B0 | B1], C_combined = [C0 | C1]
    A0_elems = M0 * K0
    A1_elems = M1 * K1
    B0_elems = K0 * N0
    B1_elems = K1 * N1
    C0_elems = M0 * N0
    C1_elems = M1 * N1

    A_combined_elems = A0_elems + A1_elems
    B_combined_elems = B0_elems + B1_elems
    C_combined_elems = C0_elems + C1_elems

    @device(AIEDevice.npu1)
    def device_body():
        tiles      = [[tile(col, row) for col in range(4)] for row in range(6)]
        shim_tiles = tiles[0]
        mem_tiles  = tiles[1]
        core_tiles = tiles[2:]

        scalar_suffix = "_scalar" if use_scalar else ""
        _m, _k, _n = prog_dims[0][3], prog_dims[0][4], prog_dims[0][5]
        C_l1_ty = np.ndarray[(_m, _n), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(_m, _k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(_k, _n), np.dtype[dtype_in]]

        zero   = external_func(f"zero{scalar_suffix}_{dtype_out_str}",   inputs=[C_l1_ty])
        matmul = external_func(f"matmul{scalar_suffix}_{dtype_in_str}_{dtype_out_str}",
                            inputs=[A_l1_ty, B_l1_ty, C_l1_ty])

        for prog_idx, (M, K, N, m, k, n) in enumerate(prog_dims):

            col_base = prog_idx * COLS_PER_PROG

            n_aie_cores      = N_AIE_ROWS * COLS_PER_PROG
            n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
            n_A_tiles_per_shim = N_AIE_ROWS // COLS_PER_PROG  # == 2

            A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
            B_l2_ty = np.ndarray[(k * n,),                      np.dtype[dtype_in]]
            C_l2_ty = np.ndarray[(m * n * N_AIE_ROWS,),         np.dtype[dtype_out]]
            A_l1_ty = np.ndarray[(m, k),  np.dtype[dtype_in]]
            B_l1_ty = np.ndarray[(k, n),  np.dtype[dtype_in]]
            C_l1_ty = np.ndarray[(m, n),  np.dtype[dtype_out]]

            A_l3l2_fifos = [None] * COLS_PER_PROG
            A_l2l1_fifos = [None] * N_AIE_ROWS
            B_l3l2_fifos = [None] * COLS_PER_PROG
            B_l2l1_fifos = [None] * COLS_PER_PROG
            C_l1l2_fifos = [[None] * COLS_PER_PROG for _ in range(N_AIE_ROWS)]
            C_l2l3_fifos = [None] * COLS_PER_PROG

            for i in range(COLS_PER_PROG):
                A_l3l2_fifos[i] = object_fifo(
                    f"A_L3L2_p{prog_idx}_{i}",
                    shim_tiles[col_base + i],
                    mem_tiles[col_base + i],
                    fifo_depth,
                    A_l2_ty,
                )

            for row in range(N_AIE_ROWS):
                A_l2l1_fifos[row] = object_fifo(
                    f"A_L2L1_p{prog_idx}_{row}",
                    mem_tiles[col_base + row // n_A_tiles_per_shim],
                    [core_tiles[row][col_base + c] for c in range(COLS_PER_PROG)],
                    fifo_depth,
                    A_l1_ty,
                    (
                        [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
                        if not use_scalar else []
                    ),
                )

            for i in range(COLS_PER_PROG):
                start_row = i * n_A_tiles_per_shim
                stop_row  = start_row + n_A_tiles_per_shim
                of_offsets = [m * k * j for j in range(n_A_tiles_per_shim)]
                object_fifo_link(
                    A_l3l2_fifos[i],
                    [A_l2l1_fifos[row] for row in range(start_row, stop_row)],
                    [],
                    of_offsets,
                )

            for c in range(COLS_PER_PROG):
                B_l3l2_fifos[c] = object_fifo(
                    f"B_L3L2_p{prog_idx}_{c}",
                    shim_tiles[col_base + c],
                    mem_tiles[col_base + c],
                    fifo_depth,
                    B_l2_ty,
                )
                B_l2l1_fifos[c] = object_fifo(
                    f"B_L2L1_p{prog_idx}_{c}",
                    mem_tiles[col_base + c],
                    [core_tiles[row][col_base + c] for row in range(N_AIE_ROWS)],
                    fifo_depth,
                    B_l1_ty,
                    (
                        (
                            [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
                            if not b_col_maj
                            else [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
                        )
                        if not use_scalar else []
                    ),
                )
                object_fifo_link(B_l3l2_fifos[c], B_l2l1_fifos[c])

            for c in range(COLS_PER_PROG):
                for row in range(N_AIE_ROWS):
                    C_l1l2_fifos[row][c] = object_fifo(
                        f"C_L1L2_p{prog_idx}_{c}_{row}",
                        core_tiles[row][col_base + c],
                        mem_tiles[col_base + c],
                        fifo_depth,
                        C_l1_ty,
                    )
                C_l2l3_fifos[c] = object_fifo(
                    f"C_L2L3_p{prog_idx}_{c}",
                    mem_tiles[col_base + c],
                    shim_tiles[col_base + c],
                    fifo_depth,
                    C_l2_ty,
                    (
                        (
                            [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
                            if not c_col_maj
                            else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
                        )
                        if not use_scalar else []
                    ),
                )
                of_offsets = [m * n * row for row in range(N_AIE_ROWS)]
                object_fifo_link(
                    [C_l1l2_fifos[row][c] for row in range(N_AIE_ROWS)],
                    C_l2l3_fifos[c],
                    of_offsets,
                    [],
                )

            all_A_l3l2_fifos[prog_idx] = A_l3l2_fifos
            all_B_l3l2_fifos[prog_idx] = B_l3l2_fifos
            all_C_l2l3_fifos[prog_idx] = C_l2l3_fifos

            def make_core_body(row, c, col_base, n_tiles_per_core, K, k,
                   A_l2l1_fifos, B_l2l1_fifos, C_l1l2_fifos, m, n):
                @core(core_tiles[row][col_base + c], f"mm_{m}x{k}x{n}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
                        for _ in loop:
                            elem_out = C_l1l2_fifos[row][c].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(K // k):
                                elem_in_a = A_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_b = B_l2l1_fifos[c].acquire(ObjectFifoPort.Consume, 1)
                                matmul(elem_in_a, elem_in_b, elem_out)
                                A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[c].release(ObjectFifoPort.Consume, 1)
                            C_l1l2_fifos[row][c].release(ObjectFifoPort.Produce, 1)

            for row in range(N_AIE_ROWS):
                for c in range(COLS_PER_PROG):
                    make_core_body(row, c, col_base, n_tiles_per_core, K, k,
                                   A_l2l1_fifos, B_l2l1_fifos, C_l1l2_fifos, m, n)

        # Runtime sequence uses combined buffers:
        #   A_combined: [A0 (M0*K0 elems) | A1 (M1*K1 elems)]
        #   B_combined: [B0 (K0*N0 elems) | B1 (K1*N1 elems)]
        #   C_combined: [C0 (M0*N0 elems) | C1 (M1*N1 elems)]
        # These are all different total sizes so aiecc.py cannot merge them.
        @runtime_sequence(
            np.ndarray[(A_combined_elems,), np.dtype[dtype_in]],   # bo0 = A_combined
            np.ndarray[(B_combined_elems,), np.dtype[dtype_in]],   # bo1 = B_combined
            np.ndarray[(C_combined_elems,), np.dtype[dtype_out]],  # bo2 = C_combined
        )
        def sequence(A_combined, B_combined, C_combined):
            # Offsets into combined buffers (in elements)
            A_offsets = [0, A0_elems]        # A0 starts at 0, A1 at A0_elems
            B_offsets = [0, B0_elems]        # B0 starts at 0, B1 at B0_elems
            C_offsets = [0, C0_elems]        # C0 starts at 0, C1 at C0_elems

            input_buffers = [
                (A_combined, B_combined, C_combined, A_offsets[0], B_offsets[0], C_offsets[0]),
                (A_combined, B_combined, C_combined, A_offsets[1], B_offsets[1], C_offsets[1]),
            ]

            tb_max_n_rows = 4 if not c_col_maj else 2

            max_tb = max(
                ceildiv(M // m // N_AIE_ROWS, tb_max_n_rows)
                for (M, K, N, m, k, n) in prog_dims
            )

            for tb in range(max_tb):
                for pingpong in [0, 1]:
                    any_bds = False
                    for prog_idx, (M, K, N, m, k, n) in enumerate(prog_dims):
                        A, B, C, A_base, B_base, C_base = input_buffers[prog_idx]
                        col_base = prog_idx * COLS_PER_PROG
                        bd_id_base = 8 * pingpong
                        n_A_tiles_per_shim = N_AIE_ROWS // COLS_PER_PROG

                        row_base  = tb * tb_max_n_rows + pingpong * (tb_max_n_rows // 2)
                        tb_n_rows = min(tb_max_n_rows // 2, M // m // N_AIE_ROWS - row_base)
                        if tb_n_rows <= 0:
                            continue
                        any_bds = True

            

                        for col in range(COLS_PER_PROG):
                            # ---- C output ----
                            if not c_col_maj:
                                C_offset = C_base + row_base * m * N_AIE_ROWS * N + col * n
                                C_sizes  = [tb_n_rows, N // n // COLS_PER_PROG, m * N_AIE_ROWS, n]
                                C_strides = [m * N_AIE_ROWS * N, n * COLS_PER_PROG, N, 1]
                            else:
                                C_offset = C_base + col * n * M + row_base * m * N_AIE_ROWS
                                C_sizes  = [N // n // COLS_PER_PROG, N_AIE_ROWS, n, m]
                                C_strides = [M * n * COLS_PER_PROG, m, M, 1]

                            npu_dma_memcpy_nd(
                                metadata=all_C_l2l3_fifos[prog_idx][col],
                                bd_id=bd_id_base,
                                mem=C,
                                offsets=[0, 0, 0, C_offset],
                                sizes=C_sizes,
                                strides=C_strides,
                            )
                            if generate_taps:
                                all_taps[prog_idx]["C"].append(
                                    TensorAccessPattern((M, N), offset=C_offset,
                                                        sizes=C_sizes, strides=C_strides)
                                )

                            for tile_row in range(tb_n_rows):
                                # ---- A input ----
                                A_block_offset = A_base + (row_base + tile_row) * N_AIE_ROWS * m * K
                                A_row_offset   = col * n_A_tiles_per_shim * m * K
                                A_offset       = A_block_offset + A_row_offset
                                A_sizes  = [N // n // COLS_PER_PROG, K // k, m * n_A_tiles_per_shim, k]
                                A_strides = [0, k, K, 1]

                                npu_dma_memcpy_nd(
                                    metadata=all_A_l3l2_fifos[prog_idx][col],
                                    bd_id=bd_id_base + 2 * tile_row + 1,
                                    mem=A,
                                    offsets=[0, 0, 0, A_offset],
                                    sizes=A_sizes,
                                    strides=A_strides,
                                )
                                if generate_taps:
                                    all_taps[prog_idx]["A"].append(
                                        TensorAccessPattern((M, K), offset=A_offset,
                                                            sizes=A_sizes, strides=A_strides)
                                    )

                                # ---- B input ----
                                B_col_offset = B_base + (col * n if not b_col_maj else col * n * K)
                                if not b_col_maj:
                                    B_sizes   = [N // n // COLS_PER_PROG, K // k, k, n]
                                    B_strides = [n * COLS_PER_PROG, k * N, N, 1]
                                else:
                                    B_sizes   = [N // n // COLS_PER_PROG, K // k, n, k]
                                    B_strides = [n * COLS_PER_PROG * K, k, K, 1]

                                npu_dma_memcpy_nd(
                                    metadata=all_B_l3l2_fifos[prog_idx][col],
                                    bd_id=bd_id_base + 2 * tile_row + 2,
                                    mem=B,
                                    offsets=[0, 0, 0, B_col_offset],
                                    sizes=B_sizes,
                                    strides=B_strides,
                                )
                                if generate_taps:
                                    all_taps[prog_idx]["B"].append(
                                        TensorAccessPattern((K, N), offset=B_col_offset,
                                                            sizes=B_sizes, strides=B_strides)
                                    )

                    if any_bds and (tb > 0 or (tb == 0 and pingpong > 0)):
                        dma_wait(*all_C_l2l3_fifos[0], *all_C_l2l3_fifos[1])

            dma_wait(*all_C_l2l3_fifos[0], *all_C_l2l3_fifos[1])

    if generate_taps:
        return tuple(
            (
                TensorAccessSequence.from_taps(all_taps[p]["A"]),
                TensorAccessSequence.from_taps(all_taps[p]["B"]),
                TensorAccessSequence.from_taps(all_taps[p]["C"]),
            )
            for p in range(N_PROGRAMS)
        )


if __name__ == "__main__":
    main()
