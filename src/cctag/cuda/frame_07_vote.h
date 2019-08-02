/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda.h>

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>

#include "onoff.h"

#include "framemeta.h"
#include "triple_point.h"
#include "edge_list.h"
#include "assist.h"

namespace cv {
    namespace cuda {
        using PtrStepSz16s = PtrStepSz<int16_t>;
        using PtrStepSz32u = PtrStepSz<uint32_t>;
        using PtrStepSz32s = PtrStepSz<int32_t>;
        using PtrStep16s = PtrStep<int16_t>;
        using PtrStep32u = PtrStep<uint32_t>;
        using PtrStep32s = PtrStep<int32_t>;
    }
};

namespace cctag {

struct Voting
{
    cv::cuda::PtrStepSz32s _d_edgepoint_index_table; // 2D plane for chaining TriplePoint coord

    void debug_download( const cctag::Parameters& params );
};

#ifndef NDEBUG
__device__
void debug_inner_test_consistency( FrameMetaPtr&                  meta,
                                   const char*                    origin,
                                   int                            p_idx,
                                   const TriplePoint*             p,
                                   cv::cuda::PtrStepSz32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> voters );
#endif // NDEBUG

}; // namespace cctag

