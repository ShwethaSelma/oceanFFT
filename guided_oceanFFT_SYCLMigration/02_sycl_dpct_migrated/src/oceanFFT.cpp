//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  FFT-based Ocean simulation
  based on original code by Yury Uralsky and Calvin Lin

  This sample demonstrates how to use CUFFT to synthesize and
  render an ocean surface in real-time.

  See Jerry Tessendorf's Siggraph course notes for more details:
  http://tessendorf.org/reports.html

  It also serves as an example of how to generate multiple vertex
  buffer streams from CUDA and render them using GLSL shaders.
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#define SYCLRT_PI_F 3.141592654f
#define SYCLRT_SQRT_HALF_F 0.707106781f

using namespace sycl;

#include <helper_cuda.h>
#include <helper_functions.h>

#include <chrono>
using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;
using float_ms = std::chrono::duration<float, ms::period>;

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

const char *sSDKsample = "SYCL FFT Ocean Simulation";

#define MAX_EPSILON 0.10f
#define THRESHOLD 0.15f
#define REFRESH_DELAY 10  // ms

////////////////////////////////////////////////////////////////////////////////
// constants

const unsigned int meshSize = 256;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

bool animate = true;

// FFT data
std::shared_ptr<oneapi::mkl::dft::descriptor<
    oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>
    fftPlan;
sycl::float2 *d_h0 = 0;  // heightfield at time 0
sycl::float2 *h_h0 = 0;
sycl::float2 *d_ht = 0;  // heightfield at time t
sycl::float2 *d_slope = 0;

// pointers to device object
float *g_hptr = NULL;
sycl::float2 *g_sptr = NULL;

// simulation parameters
const float g = 9.81f;        // gravitational constant
const float A = 1e-7f;        // wave scale factor
const float patchSize = 100;  // patch size
float windSpeed = 100.0f;
float windDir = SYCLRT_PI_F / 3.0f;
float dirDepend = 0.07f;

float animTime = 0.0f;
float prevTime = 0.0f;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsLimit = 1;  // FPS limit for sampling
unsigned int g_TotalErrors = 0;

////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <oceanFFT_kernel.cu>

extern "C" void syclGenerateSpectrumKernel(sycl::float2 *d_h0,
                                           sycl::float2 *d_ht,
                                           unsigned int in_width,
                                           unsigned int out_width,
                                           unsigned int out_height,
                                           float animTime, float patchSize);

extern "C" void syclUpdateHeightmapKernel(float *d_heightMap,
                                          sycl::float2 *d_ht,
                                          unsigned int width,
                                          unsigned int height, bool autoTest);

extern "C" void syclCalculateSlopeKernel(float *h, sycl::float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void runAutoTest();

// sycl functionality
void runSyclTest(char *exec_path);
void generate_h0(sycl::float2 *h0);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main() {
  animate = false;
  fpsLimit = frameCheckNumber;
  runAutoTest();
  exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void runAutoTest() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("Starting...\n\n");

  std::cout << "Device: "
            << q_ct1.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  dpct::device_info deviceProp;
  dev_ct1.get_device_info(deviceProp);

  printf("Compute capability %d.%d\n", deviceProp.get_major_version(),
         deviceProp.get_minor_version());

  // create FFT plan
  fftPlan = std::make_shared<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(
      std::vector<std::int64_t>{meshSize, meshSize});

  // allocate memory
  int spectrumSize = spectrumW * spectrumH * sizeof(sycl::float2);

  d_h0 = (sycl::float2 *)sycl::malloc_device(spectrumSize, q_ct1);
  h_h0 = (sycl::float2 *)malloc(spectrumSize);
  generate_h0(h_h0);

  q_ct1.memcpy(d_h0, h_h0, spectrumSize).wait();

  int outputSize = meshSize * meshSize * sizeof(sycl::float2);

  d_ht = (sycl::float2 *)sycl::malloc_device(outputSize, q_ct1);
  d_slope = (sycl::float2 *)sycl::malloc_device(outputSize, q_ct1);

  auto startDeviceTime = Time::now();

  char test_exepath[] = "./a.out";
  runSyclTest(test_exepath);

  auto stopDeviceTime = Time::now();
  auto Device_duration =
      std::chrono::duration_cast<float_ms>(stopDeviceTime - startDeviceTime)
          .count();
  printf("Processing time : %f (ms)\n", Device_duration);

  sycl::free(d_ht, q_ct1);
  sycl::free(d_slope, q_ct1);
  sycl::free(d_h0, q_ct1);
  free(h_h0);

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss() {
  float u1 = urand();
  float u2 = urand();

  if (u1 < 1e-6f) {
    u1 = 1e-6f;
  }

  return sqrtf(-2 * logf(u1)) * cosf(2 * SYCLRT_PI_F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A,
               float dir_depend) {
  float k_squared = Kx * Kx + Ky * Ky;

  if (k_squared == 0.0f) {
    return 0.0f;
  }

  // largest possible wave from constant wind of velocity v
  float L = V * V / g;

  float k_x = Kx / sqrtf(k_squared);
  float k_y = Ky / sqrtf(k_squared);
  float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

  float phillips = A * expf(-1.0f / (k_squared * L * L)) /
                   (k_squared * k_squared) * w_dot_k * w_dot_k;

  // filter out waves moving opposite to wind
  if (w_dot_k < 0.0f) {
    phillips *= dir_depend;
  }

  // damp out waves with very small length w << l
  // float w = L / 10000;
  // phillips *= expf(-k_squared * w * w);

  return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(sycl::float2 *h0) {
  for (unsigned int y = 0; y <= meshSize; y++) {
    for (unsigned int x = 0; x <= meshSize; x++) {
      float kx = (-(int)meshSize / 2.0f + x) * (2.0f * SYCLRT_PI_F / patchSize);
      float ky = (-(int)meshSize / 2.0f + y) * (2.0f * SYCLRT_PI_F / patchSize);

      float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

      if (kx == 0.0f && ky == 0.0f) {
        P = 0.0f;
      }

      float Er = gauss();
      float Ei = gauss();

      float h0_re = Er * P * SYCLRT_SQRT_HALF_F;
      float h0_im = Ei * P * SYCLRT_SQRT_HALF_F;

      int i = y * spectrumW + x;
      h0[i].x() = h0_re;
      h0[i].y() = h0_im;
    }
  }
}

void runSyclTest(char *exec_path) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  g_hptr = sycl::malloc_device<float>(meshSize * meshSize, q_ct1);
  g_sptr = sycl::malloc_device<sycl::float2>(meshSize * meshSize, q_ct1);

  // generate wave spectrum in frequency domain
  syclGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize,
                             animTime, patchSize);

  // execute inverse FFT to convert to spatial domain
  fftPlan->commit(q_ct1);
  if ((void *)d_ht == (void *)d_ht) {
    oneapi::mkl::dft::compute_backward(*fftPlan, (float *)d_ht);
  } else {
    oneapi::mkl::dft::compute_backward(*fftPlan, (float *)d_ht, (float *)d_ht);
  }

  // update heightmap values
  syclUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, true);

  {
    float *hptr = (float *)malloc(meshSize * meshSize * sizeof(float));
    q_ct1
        .memcpy((void *)hptr, (void *)g_hptr,
                meshSize * meshSize * sizeof(float))
        .wait();
    sdkDumpBin((void *)hptr, meshSize * meshSize * sizeof(float),
               "spatialDomain.bin");

    if (!sdkCompareBin2BinFloat("spatialDomain.bin", "ref_spatialDomain.bin",
                                meshSize * meshSize, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(hptr);
  }

  // calculate slope for shading
  syclCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

  {
    sycl::float2 *sptr =
        (sycl::float2 *)malloc(meshSize * meshSize * sizeof(sycl::float2));
    q_ct1
        .memcpy((void *)sptr, (void *)g_sptr,
                meshSize * meshSize * sizeof(sycl::float2))
        .wait();
    sdkDumpBin(sptr, meshSize * meshSize * sizeof(sycl::float2),
               "slopeShading.bin");

    if (!sdkCompareBin2BinFloat("slopeShading.bin", "ref_slopeShading.bin",
                                meshSize * meshSize * 2, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(sptr);
  }

  sycl::free(g_hptr, q_ct1);
  sycl::free(g_sptr, q_ct1);
}
