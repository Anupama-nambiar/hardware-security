//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <numeric>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

float abs_tol = matmul_common::get_abs_tol<C_DATATYPE>();
float rel_tol = matmul_common::get_rel_tol<C_DATATYPE>();

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);
  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();
  int c_col_maj = vm["c_col_maj"].as<int>();

  srand(1726250518);

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();
  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1)
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;

  int A_VOLUME = M * K;
  int B_VOLUME = N * K;
  int C_VOLUME = M * N;

  size_t A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  size_t B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
  size_t C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1)
                                   std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  device.register_xclbin(xclbin);

  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp1 =
      xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size * 4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    // random data : AVec[i] = matmul_common::get_random<A_DATATYPE>();
    // all zeroes  : AVec[i] = 0;
    // all ones    : AVec[i] = 1;
    // diagonal:
    if (i % N == i / N)
      AVec[i] = 0.0;
    else
      AVec[i] = 0.0;
  }
  memcpy(bufA, AVec.data(), AVec.size() * sizeof(A_DATATYPE));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    // random data : BVec[i] = matmul_common::get_random<B_DATATYPE>() * i;
    // all zeroes  : BVec[i] = 0;
    // all ones    : BVec[i] = 1;
    // diagonal:
    if (i % N == i / N)
      BVec[i] = 0.0;
    else
      BVec[i] = 0.0;
  }
  memcpy(bufB, BVec.data(), BVec.size() * sizeof(B_DATATYPE));

  char *bufOut = bo_out.map<char *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME);
  memset(bufOut, 0, C_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(AVec, K);
    std::cout << "B = \n";
    matmul_common::print_matrix(BVec, N);
  }

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;

  // Declared OUTSIDE the loop so they accumulate across all iterations
  std::vector<float> npu_times;
  std::vector<float> sync_times;
  npu_times.reserve(n_iterations);
  sync_times.reserve(n_iterations);

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out,
                      bo_tmp1, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();

    // Time the sync separately to measure host-side readback overhead
    auto sync_start = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto sync_stop = std::chrono::high_resolution_clock::now();
    float sync_time =
        std::chrono::duration_cast<std::chrono::microseconds>(sync_stop - sync_start)
            .count();

    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations)
      continue;

    // Record both npu and sync times for non-warmup iterations
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    npu_times.push_back(npu_time);
    sync_times.push_back(sync_time);

    if (do_verify) {
      memcpy(CVec.data(), bufOut, CVec.size() * sizeof(C_DATATYPE));
      if (verbosity >= 1) {
        if (do_verify_stochastic)
          std::cout << "Verifying " << verify_stochastic_n_samples
                    << " random samples against reference matmul ..." << std::endl;
        else
          std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      if (do_verify_stochastic) {
        errors = matmul_common::verify_stochastic<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verify_stochastic_n_samples, verbosity,
            abs_tol, rel_tol, b_col_maj, c_col_maj);
      } else {
        errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verbosity, abs_tol, rel_tol, b_col_maj,
            c_col_maj);
      }
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart).count();
      if (verbosity >= 1)
        std::cout << "Verify time: " << vtime << " s." << std::endl;
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }
  }

  // --- NPU timing stats ---
  double mean = std::accumulate(npu_times.begin(), npu_times.end(), 0.0)
              / npu_times.size();
  double min_time = *std::min_element(npu_times.begin(), npu_times.end());
  double max_time = *std::max_element(npu_times.begin(), npu_times.end());
  double var = 0.0;
  for (auto t : npu_times) { double d = t - mean; var += d * d; }
  var /= npu_times.size();
  double stddev = std::sqrt(var);

  std::cout << "\nAvg NPU matmul time: " << mean << "us." << std::endl;
  std::cout << "Stddev NPU matmul time: " << stddev << "us." << std::endl;
  std::cout << "Coefficient of variance NPU: " << (stddev / mean) * 100 << std::endl;
  std::cout << "Avg NPU gflops: " << macs / (1000 * mean) << std::endl;
  std::cout << "Min NPU matmul time: " << min_time << "us." << std::endl;
  std::cout << "Max NPU gflops: " << macs / (1000 * min_time) << std::endl;
  std::cout << "Max NPU matmul time: " << max_time << "us." << std::endl;
  std::cout << "Min NPU gflops: " << macs / (1000 * max_time) << std::endl;

  // --- Sync timing stats ---
  double sync_mean = std::accumulate(sync_times.begin(), sync_times.end(), 0.0)
                   / sync_times.size();
  double sync_min = *std::min_element(sync_times.begin(), sync_times.end());
  double sync_max = *std::max_element(sync_times.begin(), sync_times.end());
  double sync_var = 0.0;
  for (auto t : sync_times) { double d = t - sync_mean; sync_var += d * d; }
  sync_var /= sync_times.size();
  double sync_stddev = std::sqrt(sync_var);

  std::cout << "\nAvg sync (readback) time: " << sync_mean << "us." << std::endl;
  std::cout << "Stddev sync time: " << sync_stddev << "us." << std::endl;
  std::cout << "Min sync time: " << sync_min << "us." << std::endl;
  std::cout << "Max sync time: " << sync_max << "us." << std::endl;

  if (trace_size > 0)
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors;
    if (do_verify_stochastic)
      std::cout << " (out of " << verify_stochastic_n_samples << " random samples)";
    std::cout << "\n\nFailed.\n\n";
    return 1;
  }
}
