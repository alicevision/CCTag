/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "onoff.h"

#include <cctag/cuda/cctag_cuda_runtime.h>

#include "frame.h"

using namespace std;

// NOTE
// Excellent easy speedup possible by using threadIdx.x for parallel checking
// of the voter array in count_winners!!!

#define CONC_POINTS 8

namespace cctag {

namespace vote {

__device__ inline
void count_winners( FrameMetaPtr&                   meta,
                    const int                       offset,
                    const int                       maxpoints,
                    const int*                      inner_points,
                    const DevEdgeList<TriplePoint>& voters,
                    const float*                    chosen_flow_length,
                    const DevEdgeList<int>          chosen_idx )
{
    int          inner_point_index[CONC_POINTS];
    TriplePoint* inner_point[CONC_POINTS];
    int          winner_size[CONC_POINTS];
    float        flow_length[CONC_POINTS];

    for( int point=0; point<CONC_POINTS; point++ ) {
        const bool pred          = ( offset+point < maxpoints );
        const int  ipi           = pred ? inner_points[offset+point] : 0;
        inner_point_index[point] = ipi;
        inner_point[point]       = pred ? &voters.ptr[ipi] : 0;
        winner_size[point]       = 0;
        flow_length[point]       = 0.0f;
    }

    /* This loop looks dangerous, but it is actually faster than
     * a manually partially unrolled loop.
     */
    const int voter_list_size = meta.list_size_voters();
    const int start_offset = 32 * threadIdx.y + threadIdx.x;
    const int inc_offset   = 32*32;
    for( int i=start_offset; i<voter_list_size; i+=inc_offset ) {
        const int my_vote = chosen_idx.ptr[i];
        for( int point=0; point<CONC_POINTS; point++ ) {
            if( inner_point[point] == 0 ) continue;

            if( my_vote == inner_point_index[point] ) {
                winner_size[point] += 1;
                flow_length[point] += chosen_flow_length[i];
            }
        }
    }

    for( int point=0; point<CONC_POINTS; point++ ) {
        if( inner_point[point] == 0 ) continue;

        winner_size[point] += cctag::shuffle_down( winner_size[point], 16 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  8 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  4 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  2 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  1 );
        winner_size[point]  = cctag::shuffle     ( winner_size[point],  0 );

        flow_length[point] += cctag::shuffle_down( flow_length[point], 16 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  8 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  4 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  2 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  1 );
        flow_length[point]  = cctag::shuffle     ( flow_length[point],  0 );
    }

    __shared__ int   winner_array[CONC_POINTS][32];
    __shared__ float length_array[CONC_POINTS][32];

    if( threadIdx.x < CONC_POINTS ) {
        const int point = threadIdx.x;

        winner_array[point][threadIdx.y] = winner_size[point];
        length_array[point][threadIdx.y] = flow_length[point];
    }

    __syncthreads();

    if( threadIdx.y < CONC_POINTS ) {
        const int point = threadIdx.y;

        if( inner_point[point] == 0 ) return;

        winner_size[point] = winner_array[point][threadIdx.x];
        flow_length[point] = length_array[point][threadIdx.x];

        winner_size[point] += cctag::shuffle_down( winner_size[point], 16 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  8 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  4 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  2 );
        winner_size[point] += cctag::shuffle_down( winner_size[point],  1 );

        flow_length[point] += cctag::shuffle_down( flow_length[point], 16 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  8 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  4 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  2 );
        flow_length[point] += cctag::shuffle_down( flow_length[point],  1 );

        if( threadIdx.x == 0 ) {
            inner_point[point]->_winnerSize = winner_size[point];
            inner_point[point]->_flowLength = flow_length[point] / winner_size[point];
        }
    }
}

/* For all chosen inner points, compute the average flow length and the
 * number of voters, and store in the TriplePoint structure of the chosen
 * inner point.
 *
 * voters is the list of all edges with their chaining info.
 * inner_points is a list of indices into that list, containing the sorted,
 * unique indices of chosen inner points.
 */
__global__
void eval_chosen( FrameMetaPtr             meta,
                  DevEdgeList<TriplePoint> voters,       // input-output
                  const float*             chosen_flow_length, // input
                  DevEdgeList<int>         chosen_idx, // input
                  DevEdgeList<int>         inner_points // input
                )
{
    uint32_t offset = blockIdx.x * CONC_POINTS;
    const int maxpoints = meta.list_size_inner_points();
    if( offset >= maxpoints ) {
        return;
    }

    vote::count_winners( meta,
                         offset,
                         maxpoints,
                         inner_points.ptr,
                         voters,
                         chosen_flow_length,
                         chosen_idx );
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION_FOR_EVAL

namespace vote
{

__global__
void dp_call_eval_chosen( FrameMetaPtr             meta,
                          DevEdgeList<TriplePoint> voters, // modified input
                          const float*             chosen_flow_length, // input
                          DevEdgeList<int>         chosen_idx,
                          DevEdgeList<int>         inner_points ) // input
{
    int listsize = meta.list_size_interm_inner_points();

    dim3 block( 32, 32, 1 );
    dim3 grid ( grid_divide( listsize, CONC_POINTS ), 1, 1 );

    vote::eval_chosen
        <<<grid,block>>>
        ( meta,
          voters,
          chosen_flow_length,
          chosen_idx,
          inner_points );
}

} // namespace vote

__host__
bool Frame::applyVoteEval( )
{
#ifndef NDEBUG
    _interm_inner_points.copySizeFromDevice( _stream, EdgeListCont );
    POP_CHK_CALL_IFSYNC;
    _voters.copySizeFromDevice( _stream, EdgeListWait );
    cerr << "Debug voting (with separable compilation)"
         << " # seed indices 2: " << _interm_inner_points.host.size
         << " # chained edgeco: " << _voters.host.size << endl;
#endif

    POP_CHK_CALL_IFSYNC;
    vote::dp_call_eval_chosen
        <<<1,1,0,_stream>>>
        ( _meta,
          _voters.dev,
          _v_chosen_flow_length,
          _v_chosen_idx.dev,
          _interm_inner_points.dev );
    POP_CHK_CALL_IFSYNC;
    return true;
}

#else // not USE_SEPARABLE_COMPILATION_FOR_EVAL
__host__
bool Frame::applyVoteEval( )
{
    /* Without Dynamic Parallelism, we must block here to retrieve the
     * value d_num_selected_out from the device before the voting
     * step.
     */
// #ifndef NDEBUG
#if 0
    _voters.copySizeFromDevice( _stream, EdgeListCont );
#endif
    _interm_inner_points.copySizeFromDevice( _stream, EdgeListWait );

// #ifndef NDEBUG
#if 0
    cerr << "Debug voting (without separable compilation)"
         << " # inner points: " << _interm_inner_points.host.size
         << " # voters      : " << _voters.host.size << endl;
#endif

    /* Add number of voters to chosen inner points, and
     * add average flow length to chosen inner points.
     */
    dim3 block( 32, 32, 1 );
    dim3 grid ( grid_divide( _interm_inner_points.host.size, CONC_POINTS ), 1, 1 );

    vote::eval_chosen
        <<<grid,block,0,_stream>>>
        ( _meta,
          _voters.dev,
          _v_chosen_flow_length,
          _v_chosen_idx.dev,
          _interm_inner_points.dev );
    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_EVAL

} // namespace cctag

