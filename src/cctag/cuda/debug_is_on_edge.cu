/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/cctag_cuda_runtime.h>

#include "debug_is_on_edge.h"
#include "assist.h"

namespace cctag
{

using namespace std;

#ifndef NDEBUG
__global__
void debug_point_is_on_edge( FrameMetaPtr         meta,
                             cv::cuda::PtrStepSzb edge_img,
                             DevEdgeList<short2>  all_edgecoords )
{
    int offset = blockIdx.x * 32 + threadIdx.x;
    if( offset >= meta.list_size_all_edgecoords() ) return;
    short2& coord = all_edgecoords.ptr[offset];
    assert( coord.x > 0 );
    assert( coord.y > 0 );
    assert( coord.x < edge_img.cols );
    assert( coord.y < edge_img.rows );
    assert( edge_img.ptr(coord.y)[coord.x] == 1 );
}

__host__
void debugPointIsOnEdge( FrameMetaPtr&               meta,
                         const cv::cuda::PtrStepSzb& edge_img,
                         const EdgeList<short2>&     all_edgecoords,
                         cudaStream_t                stream )
{
    // cerr << "  Enter " << __FUNCTION__ << endl;

    int sz;
    meta.fromDevice( List_size_all_edgecoords, sz, stream );
    POP_CUDA_SYNC( stream );
    // cerr << "    Listlength " << sz << endl;
    if( sz == 0 ) {
        // cerr << "  Leave " << __FUNCTION__ << endl;
        return;
    }

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = grid_divide( sz, 32 );
    debug_point_is_on_edge
        <<<grid,block,0,stream>>>
        ( meta,
          edge_img,
          all_edgecoords.dev );

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( stream );
    // cerr << "  Leave " << __FUNCTION__ << endl;
}
#endif // NDEBUG

}; // namespace cctag

