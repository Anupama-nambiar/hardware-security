#parallel mult 
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array
targetname=whole_array_parallel

M?=512
K?=512
N?=512
m?=64
k?=64
n?=32
n_aie_cols?=2
b_col_maj?=0
c_col_maj?=0
col_offset?=0

kernels=mm_${m}x${k}x${n}

aieargs+=-m ${m} -k ${k} -n ${n} --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj}
runargs+=--iters 30 --warmup 15 --verbosity 1 --verify 1

target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

use_scalar?=0
use_linear_buf_alloc?=1
buffer_aloc_flag=basic-sequential

ifeq (${c_col_maj}, 1)
aieargs+=--c-col-maj ${c_col_maj}
runargs+=--c_col_maj ${c_col_maj}

endif

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n}

ifeq (${b_col_maj}, 1)
KERNEL_DEFINES+=-DB_COL_MAJ
endif
ifeq (${c_col_maj}, 1)
KERNEL_DEFINES+=-DC_COL_MAJ
endif
ifeq (${use_scalar}, 1)
KERNEL_DEFINES+=-DSCALAR_ONLY
aieargs+=--scalar 1
else
KERNEL_DEFINES+=-DVECTORIZED_ONLY
endif

include ${srcdir}/../makefile-common

# -------------------------------------------------------
# Dual-program xclbin: both programs share one Python file
# that emits a single AIE module using both column groups.
# -------------------------------------------------------
dual_xclbin_target=build/final_dual_${target_suffix}.xclbin
dual_mlir_target=build/aie_dual_${target_suffix}.mlir
dual_insts_target=build/insts_dual_${target_suffix}.txt

# The dual design Python script receives the same aieargs but
# uses --n-aie-cols 2 (each program gets 2 cols of the 4-col array).
dual_aieargs=${aieargs}

${dual_mlir_target}: ${srcdir}/${targetname}_dual.py
	mkdir -p ${@D}
	python3 $< ${dual_aieargs} --trace_size 0 > $@ || \
	    (echo "Dual MLIR generation failed" && rm -f $@ && exit 1)

${dual_xclbin_target}: ${dual_mlir_target} build/mm_${m}x${k}x${n}.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py \
	    --alloc-scheme=${buffer_aloc_flag} \
	    --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=${@F} \
	    ${aiecc_peano_flags} \
	    --aie-generate-npu-insts \
	    --npu-insts-name=${dual_insts_target:build/%=%} \
	    $(<:%=../%)

# Kernel object (shared by both programs - same tile dims)
build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}
