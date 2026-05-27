//===- test_parallel.cpp ----------------------------------000---*- C++ -*-===//
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
#ifndef DTYPE_IN
#define DTYPE_IN uint16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT uint16_t
#endif
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

float abs_tol = 1.0f;
float rel_tol = 0.05f;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);
  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity           = vm["verbosity"].as<int>();
  int do_verify           = vm["verify"].as<bool>();
  int n_iterations        = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int b_col_maj           = vm["b_col_maj"].as<int>();
  int c_col_maj           = vm["c_col_maj"].as<int>();

  srand(1726250518);
  std::cout << verbosity << std::endl;

  int M = vm["M"].as<int>(), K = vm["K"].as<int>(), N = vm["N"].as<int>();
  int M1 = M, K1 = K, N1 = N;

  bool do_verify_stochastic = (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1)
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;

  int A0_ELEMS = M * K;
  int A1_ELEMS = M * K;
  int B0_ELEMS = N * K;
  int B1_ELEMS = N * K;
  int C0_ELEMS = M * N;
  int C1_ELEMS = M * N;

  size_t A_COMBINED_SIZE = (A0_ELEMS + A1_ELEMS) * sizeof(A_DATATYPE);
  size_t B_COMBINED_SIZE = (B0_ELEMS + B1_ELEMS) * sizeof(B_DATATYPE);
  size_t C_COMBINED_SIZE = (C0_ELEMS + C1_ELEMS) * sizeof(C_DATATYPE);

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
  auto xkernel  = *std::find_if(xkernels.begin(), xkernels.end(),
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

  if (verbosity >= 1) std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a_combined = xrt::bo(device, A_COMBINED_SIZE,
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b_combined = xrt::bo(device, B_COMBINED_SIZE,
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c_combined = xrt::bo(device, C_COMBINED_SIZE,
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp1 =
      xrt::bo(device, sizeof(int), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace =
      xrt::bo(device, sizeof(int), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  A_DATATYPE *bufA_combined = bo_a_combined.map<A_DATATYPE *>();
  B_DATATYPE *bufB_combined = bo_b_combined.map<B_DATATYPE *>();
  C_DATATYPE *bufC_combined = bo_c_combined.map<C_DATATYPE *>();

  A_DATATYPE *bufA0 = bufA_combined;
  A_DATATYPE *bufA1 = bufA_combined + A0_ELEMS;
  B_DATATYPE *bufB0 = bufB_combined;
  B_DATATYPE *bufB1 = bufB_combined + B0_ELEMS;
  C_DATATYPE *bufC0 = bufC_combined;
  C_DATATYPE *bufC1 = bufC_combined + C0_ELEMS;

  std::vector<A_DATATYPE> AVec(A0_ELEMS), AVec1(A1_ELEMS);
  // Diagonal (change to all zeroes or all ones as needed)
  for (int i = 0; i < A0_ELEMS; i++)
    AVec[i]  = (i % N == i / N) ? 1.0 : 0.0;
  for (int i = 0; i < A1_ELEMS; i++)
    AVec1[i] = (i % N == i / N) ? 1.0 : 0.0;
  memcpy(bufA0, AVec.data(),  A0_ELEMS * sizeof(A_DATATYPE));
  memcpy(bufA1, AVec1.data(), A1_ELEMS * sizeof(A_DATATYPE));

  std::vector<B_DATATYPE> BVec(B0_ELEMS), BVec1(B1_ELEMS);
  // Diagonal (change to all zeroes or all ones as needed)
  for (int i = 0; i < B0_ELEMS; i++)
    BVec[i]  = (i % N == i / N) ? 1.0 : 0.0;
  for (int i = 0; i < B1_ELEMS; i++)
    BVec1[i] = (i % N == i / N) ? 1.0 : 0.0;
  memcpy(bufB0, BVec.data(),  B0_ELEMS * sizeof(B_DATATYPE));
  memcpy(bufB1, BVec1.data(), B1_ELEMS * sizeof(B_DATATYPE));

  memset(bufC0, 0, C0_ELEMS * sizeof(C_DATATYPE));
  memset(bufC1, 0, C1_ELEMS * sizeof(C_DATATYPE));

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
  }

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a_combined.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b_combined.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c_combined.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_tmp1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;

  // Declared OUTSIDE the loop so they accumulate across all iterations
  std::vector<float> npu_times;
  std::vector<float> sync_times;
  npu_times.reserve(n_iterations);
  sync_times.reserve(n_iterations);

  int errors = 0;
  float macs = 2.0f * 2.0f * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iter << ").\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(),
                      bo_a_combined, bo_b_combined, bo_c_combined,
                      bo_tmp1, bo_trace);

    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    auto stop = std::chrono::high_resolution_clock::now();

    // Time the sync separately to measure host-side readback overhead
    auto sync_start = std::chrono::high_resolution_clock::now();
    bo_c_combined.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto sync_stop = std::chrono::high_resolution_clock::now();
    float sync_time =
        std::chrono::duration_cast<std::chrono::microseconds>(sync_stop - sync_start)
            .count();

    if (iter < n_warmup_iterations)
      continue;

    // Record both npu and sync times for non-warmup iterations
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    npu_times.push_back(npu_time);
    sync_times.push_back(sync_time);

    if (do_verify) {
      std::vector<C_DATATYPE> CVec(C0_ELEMS);
      memcpy(CVec.data(), bufC0, C0_ELEMS * sizeof(C_DATATYPE));
      int errors0 = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
          M, N, K, AVec, BVec, CVec, verbosity, abs_tol, rel_tol, b_col_maj, c_col_maj);
      std::cout << "Program 0 errors: " << errors0 << "\n";

      std::vector<C_DATATYPE> CVec1(C1_ELEMS);
      memcpy(CVec1.data(), bufC1, C1_ELEMS * sizeof(C_DATATYPE));
      int errors1 = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
          M, N, K, AVec1, BVec1, CVec1, verbosity, abs_tol, rel_tol, b_col_maj, c_col_maj);
      std::cout << "Program 1 errors: " << errors1 << "\n";

      errors = errors0 + errors1;

      if (verbosity >= 1) {
        if (do_verify_stochastic)
          std::cout << "Verifying " << verify_stochastic_n_samples
                    << " random samples against reference matmul ...\n";
        else
          std::cout << "Verifying against reference matmul ...\n";
      }

      auto vstart = std::chrono::system_clock::now();
      if (do_verify_stochastic) {
        errors = matmul_common::verify_stochastic<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verify_stochastic_n_samples,
            verbosity, abs_tol, rel_tol, b_col_maj, c_col_maj);
      } else {
        errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
            M, N, K, AVec, BVec, CVec, verbosity, abs_tol, rel_tol,
            b_col_maj, c_col_maj);
      }
      auto vstop = std::chrono::system_clock::now();
      if (verbosity >= 1) {
        float vtime = std::chrono::duration_cast<std::chrono::seconds>(
                          vstop - vstart).count();
        std::cout << "Verify time: " << vtime << " s.\n";
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified.\n";
    }
  }

  if (npu_times.empty()) {
    std::cout << "No timed iterations recorded.\n";
    return 0;
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

  std::cout << "\nAvg NPU matmul time: " << mean << "us.\n";
  std::cout << "Stddev NPU matmul time: " << stddev << "us.\n";
  std::cout << "Coefficient of variance: " << (stddev / mean) * 100 << "\n";
  std::cout << "Avg NPU gflops: " << macs / (1000 * mean) << "\n";
  std::cout << "Min NPU matmul time: " << min_time << "us.\n";
  std::cout << "Max NPU gflops: " << macs / (1000 * min_time) << "\n";
  std::cout << "Max NPU matmul time: " << max_time << "us.\n";
  std::cout << "Min NPU gflops: " << macs / (1000 * max_time) << "\n";

  // --- Sync timing stats ---
  double sync_mean = std::accumulate(sync_times.begin(), sync_times.end(), 0.0)
                   / sync_times.size();
  double sync_min = *std::min_element(sync_times.begin(), sync_times.end());
  double sync_max = *std::max_element(sync_times.begin(), sync_times.end());
  double sync_var = 0.0;
  for (auto t : sync_times) { double d = t - sync_mean; sync_var += d * d; }
  sync_var /= sync_times.size();
  double sync_stddev = std::sqrt(sync_var);

  std::cout << "\nAvg sync (readback) time: " << sync_mean << "us.\n";
  std::cout << "Stddev sync time: " << sync_stddev << "us.\n";
  std::cout << "Min sync time: " << sync_min << "us.\n";
  std::cout << "Max sync time: " << sync_max << "us.\n";

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
