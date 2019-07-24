/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/cctag_cuda_runtime.h>

#include "frame.h"

namespace cctag
{

#ifndef NDEBUG

// Called at the end of applyThinning to ensure that all coordinates
// in the edge list are also in the edge image.
// Called again in gradientDescent because the same data is suddently
// wrong?

__host__
void debugPointIsOnEdge( FrameMetaPtr&               meta,
                         const cv::cuda::PtrStepSzb& edge_img,
                         const EdgeList<short2>&     all_edgecoords,
                         cudaStream_t                stream );

#endif // NDEBUG

}; // namespace cctag

