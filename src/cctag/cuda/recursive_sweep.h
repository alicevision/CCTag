/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/cctag_cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"

using namespace std;

namespace cctag
{

namespace recursive_sweep
{

__host__
void expandEdges( cv::cuda::PtrStepSz32s& img, int* dev_counter, cudaStream_t stream );

__host__
void connectComponents( cv::cuda::PtrStepSz32s& img, int* dev_counter, cudaStream_t stream );

}; // namespace recursive_sweep
}; // namespace cctag

