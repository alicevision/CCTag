/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda.h>

#include "cctag/cuda/cctag_cuda_runtime.h"
#include "cctag/cuda/onoff.h"
#include "cctag/cuda/ptrstep.h"
#include "cctag/cuda/assist.h"
#include "cctag/cuda/framemeta.h"
#include "cctag/cuda/triple_point.h"
#include "cctag/cuda/edge_list.h"

namespace cctag {

struct Voting
{
    DevPlane2D32s _d_edgepoint_index_table; // 2D plane for chaining TriplePoint coord

    void debug_download( const cctag::Parameters& params );
};

#ifndef NDEBUG
__device__
void debug_inner_test_consistency( FrameMetaPtr&                  meta,
                                   const char*                    origin,
                                   int                            p_idx,
                                   const TriplePoint*             p,
                                   DevPlane2D32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> voters );
#endif // NDEBUG

}; // namespace cctag

