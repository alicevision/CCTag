#include "onoff.h"

#include <cuda_runtime.h>

#include "frame.h"
#include "assist.h"

using namespace std;

// NOTE
// Excellent easy speedup possible by using threadIdx.x for parallel checking
// of the voter array in count_winners!!!

#define CONC_POINTS 8

namespace popart {

namespace vote {

__device__ inline
void count_winners( FrameMetaPtr&              meta,
                    DevEdgeList<CudaEdgePoint> d_edgepoints, // output
                    const int                  offset,
                    const int                  maxpoints,
                    const int*                 d_inner_points,
                    // const DevEdgeList<int>&    voters,
                    const DevEdgeList<float>&  vote_weight, // input
                    const DevEdgeList<int>&    voting_for ) // input
{
    // int   inner_point_index[CONC_POINTS];
    // int*  inner_point[CONC_POINTS];
    int   inner_point[CONC_POINTS];
    int   winner_size[CONC_POINTS];
    float flow_length[CONC_POINTS];

    for( int point=0; point<CONC_POINTS; point++ ) {
        const bool pred          = ( offset+point < maxpoints );

        // Note that d_inner_points contains indices into the d_edgepoints
        // array. The same is true for voting_for. That is what we need.
        // It avoids additional indirections.
        const int  ipi           = pred ? d_inner_points[offset+point] : 0;

        // inner_point_index[point] = ipi;
        // inner_point[point]       = pred ? &voters.ptr[ipi] : 0;

        // If this loop instance is out of bounds, we re-check inner
        // point 0 (possibly several times). Avoids branches.
        inner_point[point]       = ipi;
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
        const int vote_for = voting_for.ptr[i];

        // If vote_for==-1, all threads will skip. No branching.
        if( vote_for == -1 ) continue;

        for( int point=0; point<CONC_POINTS; point++ ) {
            // if( inner_point[point] == 0 ) continue;
            // if( vote_for == inner_point_index[point] )

            // precompute whether this thread has found a vote
            bool pred_any_work = ( vote_for == inner_point[point] );

            // only do something if any threads found a vote
            // no branching
            if( __any( pred_any_work ) ) {
                // no overhead: all read the same float
                float fl = vote_weight.ptr[i];

                // increase only for the single thread that has
                // found a vote, without branching
                winner_size[point] += ( pred_any_work ? 1  : 0 );
                flow_length[point] += ( pred_any_work ? fl : 0 );
            }
        }
    }

    // We have 32x32 threads. Sum up the votes found by warp first,
    // ie. 32 in X direction
    for( int point=0; point<CONC_POINTS; point++ ) {
        winner_size[point] += __shfl_down( winner_size[point], 16 );
        winner_size[point] += __shfl_down( winner_size[point],  8 );
        winner_size[point] += __shfl_down( winner_size[point],  4 );
        winner_size[point] += __shfl_down( winner_size[point],  2 );
        winner_size[point] += __shfl_down( winner_size[point],  1 );
        winner_size[point]  = __shfl     ( winner_size[point],  0 );

        flow_length[point] += __shfl_down( flow_length[point], 16 );
        flow_length[point] += __shfl_down( flow_length[point],  8 );
        flow_length[point] += __shfl_down( flow_length[point],  4 );
        flow_length[point] += __shfl_down( flow_length[point],  2 );
        flow_length[point] += __shfl_down( flow_length[point],  1 );
        flow_length[point]  = __shfl     ( flow_length[point],  0 );
    }

    // The first CONC_POINTS threads of each warp write the
    // CONC_POINTS summations into the shared arrays
    __shared__ int   winner_array[CONC_POINTS][32];
    __shared__ float length_array[CONC_POINTS][32];

    if( threadIdx.x < CONC_POINTS ) {
        const int point = threadIdx.x;

        winner_array[point][threadIdx.y] = winner_size[point];
        length_array[point][threadIdx.y] = flow_length[point];
    }

    // hard sync to wait for __shared__ writing of all warps
    __syncthreads();

    // Now, we must sum the Y direction of the 32x32 partial
    // summations. We use one warp for each summations.
    // But only CONC_POINTS are required.
    //
    // Finally, the first thread of each warp writes the number
    // of votes and the average flow length into the edge point
    // that has received all the votes.
    //
    // Note that there is one block that will write the of point
    // 0 several times.
    if( threadIdx.y < CONC_POINTS ) {
        const int point = threadIdx.y;

        winner_size[point] = winner_array[point][threadIdx.x];
        flow_length[point] = length_array[point][threadIdx.x];

        winner_size[point] += __shfl_down( winner_size[point], 16 );
        winner_size[point] += __shfl_down( winner_size[point],  8 );
        winner_size[point] += __shfl_down( winner_size[point],  4 );
        winner_size[point] += __shfl_down( winner_size[point],  2 );
        winner_size[point] += __shfl_down( winner_size[point],  1 );

        flow_length[point] += __shfl_down( flow_length[point], 16 );
        flow_length[point] += __shfl_down( flow_length[point],  8 );
        flow_length[point] += __shfl_down( flow_length[point],  4 );
        flow_length[point] += __shfl_down( flow_length[point],  2 );
        flow_length[point] += __shfl_down( flow_length[point],  1 );

        if( threadIdx.x == 0 ) {
            // const int edge_index = *( inner_point[point] );
            const int edge_index = inner_point[point];
            CudaEdgePoint& ep = d_edgepoints.ptr[ edge_index ];
            ep._numVotes = winner_size[point];
            const float fl = flow_length[point] / winner_size[point];
            ep._avgFlowLength = fl;
        }
    }
}

/* For all chosen inner points, compute the average flow length and the
 * number of voters, and store in the TriplePoint structure of the chosen
 * inner point.
 *
 * voters is the list of all edges with their chaining info.
 * d_inner_points is a list of indices into that list, containing the sorted,
 * unique indices of chosen inner points.
 */
__global__
void eval_chosen( FrameMetaPtr       meta,
                  DevEdgeList<CudaEdgePoint> d_edgepoints, // input-output
                  // DevEdgeList<int>   voters,       // input-output
                  DevEdgeList<float> vote_weight, // input
                  DevEdgeList<int>   voting_for, // input
                  DevEdgeList<int>   d_inner_points // input
                )
{
    uint32_t offset = blockIdx.x * CONC_POINTS;
    const int maxpoints = meta.list_size_inner_points();
    if( offset >= maxpoints ) {
        return;
    }

    vote::count_winners( meta,
                         d_edgepoints,
                         offset,
                         maxpoints,
                         d_inner_points.ptr,
                         // voters,
                         vote_weight,
                         voting_for );
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION_FOR_EVAL

namespace vote
{

__global__
void dp_call_eval_chosen( FrameMetaPtr               meta,
                          DevEdgeList<CudaEdgePoint> d_edgepoints, // input-output
                          DevEdgeList<int>           voters, // input
                          DevEdgeList<float>         vote_weight, // input
                          DevEdgeList<int>           voting_for,         // input
                          DevEdgeList<int>           inner_points )      // input
{
    /* At this point, we have applied sort and unique to inner_points,
     * so that it contains all indices of edge points that have received
     * at least one vote.
     * Now, for every inner point, we must loop through voters.
     * It is irrelevant who has voted, we require only the number of voters,
     * the target of the vote is stored in voting_for, the flow length
     * for the vote is stored in vote_weight.
     *
     * For every voter v, we must check if voting_for[v] is >= 0. If it
     * is >=0, we increase the number of votes received, and we also
     * increase flowLength by vote_weight[v].
     * Finally, we divide vote_weight[v] by the number of votes received
     * by v.
     *
     * The offsets of voters, vote_weight and voting_for are identical for
     * every lookup, and many of them are unused, so many checks will be
     * skipped.
     *
     * We let every thread block process CONC_POINTS inner points in
     * parallel.
     *
     * 32x32 threads cooperate to go through the list of voters and
     * collect voting information for all of the CONC_POINTS that they
     * handle. See counter_winners() for details.
     */
    int listsize = meta.list_size_interm_inner_points();

    dim3 block( 32, 32, 1 );
    dim3 grid ( grid_divide( listsize, CONC_POINTS ), 1, 1 );

    vote::eval_chosen
        <<<grid,block>>>
        ( meta,
          d_edgepoints,
          // voters,
          vote_weight,
          voting_for,
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
          _edgepoints.dev,
          _voters.dev,
          _vote_weight.dev,
          _voting_for.dev,
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
          _edgepoints.dev,
          _voters.dev,
          _vote_weight.dev,
          _voting_for.dev,
          _interm_inner_points.dev );
    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_EVAL

} // namespace popart

