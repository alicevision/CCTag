/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <algorithm>
#include <limits>
#include <cctag/cuda/cctag_cuda_runtime.h>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace cctag
{

#if 0
__host__
void Frame::applyVote( )
{
    bool success;

    /* For every potential outer ring point in _voters,
     * check whether it votes for a point, and if yes:
     * (a) add that point to _inner_points, 
     * (b) store the index that point (value it edgepoint_index_table)
     *     in the outer ring point's TriplePoint structure
     * (c) store the flow value there as well.
     */
    success = applyVoteConstructLine( );

    if( success ) {
        /* Apply sort and uniq to the list of potential inner ring
         * points in _inner_points. Store the result in _interm_inner_points.
         */
        success = applyVoteSortUniq();

        /* For all potential inner points in _interm_inner_points,
         * count the number of voters and compute the average
         * flow size. Annotate inner points.
         */
        applyVoteEval();

        /* For all inner points in _interm_inner_points, check if
         * average flow size exceeds threshold, store all successful
         * inner point in _inner_points.
         */
        applyVoteIf();

        /* would it be better to remove unused voters from the chaincoords ? */
    }

    if( ! success ) {
        _inner_points.host.size = 0;
        _voters.host.size       = 0;
        return;
    }

    // Called separately from tag.cu
    // applyVoteDownload( );
}
#endif // 0

#ifndef NDEBUG
__device__
void debug_inner_test_consistency( FrameMetaPtr&                  meta,
                                   const char*                    origin,
                                   int                            p_idx,
                                   const TriplePoint*             p,
                                   cv::cuda::PtrStepSz32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> voters )
{
    if( p == 0 ) {
        printf("%s Impossible bug, initialized from memory address\n", origin);
        assert( 0 );
    }

    if( outOfBounds( p->coord, edgepoint_index_table ) ) {
        printf("%s Index (%d,%d) does not fit into coord lookup tables\n", origin, p->coord.x, p->coord.y );
        assert( 0 );
    }

    int idx = edgepoint_index_table.ptr(p->coord.y)[p->coord.x];
    if( idx < 0 || idx >= meta.list_size_voters() ) {
        printf("%s Looked up index (coord) is out of bounds\n", origin);
        assert( 0 );
    }

    if( idx != p_idx ) {
        printf("%s Looked up index %d is not identical to input index %d\n", origin, idx, p_idx);
        assert( 0 );
    }

    if( outOfBounds( p->descending.befor, edgepoint_index_table ) ) {
        printf("%s Before coordinates (%d,%d) do not fit into lookup tables\n", origin, p->descending.befor.x, p->descending.befor.y );
        assert( 0 );
    }

    if( outOfBounds( p->descending.after, edgepoint_index_table ) ) {
        printf("%s After coordinates (%d,%d) for [%d]=(%d,%d) do not fit into lookup tables\n", origin, p->descending.after.x, p->descending.after.y, p_idx, p->coord.x, p->coord.y );
        assert( 0 );
    }
}
#endif // NDEBUG

} // namespace cctag

