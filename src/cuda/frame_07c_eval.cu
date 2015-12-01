#include "onoff.h"

#include <cuda_runtime.h>

#include "frame.h"
// #include "assist.h"

using namespace std;

// NOTE
// Excellent easy speedup possible by using threadIdx.x for parallel checking
// of the voter array in count_winners!!!

namespace popart {

namespace vote {

__device__ inline
int count_winners( FrameMetaPtr&                   meta,
                   const int                       inner_point_index,
                   TriplePoint*                    inner_point,
                   const DevEdgeList<TriplePoint>& voters )
{
    int   winner_size = 0;
    float flow_length = 0.0f;

    /* This loop looks dangerous, but it is actually faster than
     * a manually partially unrolled loop.
     */
    const int voter_list_size = meta.list_size_voters();
    for( int i=0; i<voter_list_size; i++ ) {
        if( voters.ptr[i].my_vote == inner_point_index ) {
            winner_size += 1;
            flow_length += voters.ptr[i].chosen_flow_length;
        }
    }
    inner_point->_winnerSize = winner_size;
    inner_point->_flowLength = flow_length / winner_size;
    return winner_size;
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
                  DevEdgeList<TriplePoint> voters,      // input-output
                  DevEdgeList<int>         inner_points // input
                )
{
    uint32_t offset = threadIdx.x + blockIdx.x * 32;
    if( offset >= meta.list_size_inner_points() ) {
        return;
    }

    const int    inner_point_index = inner_points.ptr[offset];
    TriplePoint* inner_point = &voters.ptr[inner_point_index];
    vote::count_winners( meta, inner_point_index, inner_point, voters );
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION_FOR_EVAL

namespace vote
{

__global__
void dp_call_eval_chosen( FrameMetaPtr             meta,
                          DevEdgeList<TriplePoint> voters, // modified input
                          DevEdgeList<int>         inner_points )      // input
{
    int listsize = meta.list_size_inner_points();

    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( listsize, 32 ), 1, 1 );

    vote::eval_chosen
        <<<grid,block>>>
        ( meta,
          voters,
          inner_points );
}

} // namespace vote

__host__
bool Frame::applyVoteEval( )
{
    _interm_inner_points.copySizeFromDevice( _stream, EdgeListCont );

#ifndef NDEBUG
    _voters.copySizeFromDevice( _stream, EdgeListWait );
    cerr << "Debug voting (with separable compilation)"
         << " # seed indices 2: " << _interm_inner_points.host.size
         << " # chained edgeco: " << _voters.host.size << endl;
#endif

    vote::dp_call_eval_chosen
        <<<1,1,0,_stream>>>
        ( _meta,
          _voters.dev,  // output
          _interm_inner_points.dev );    // buffer
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
#ifndef NDEBUG
    _voters.copySizeFromDevice( _stream, EdgeListCont );
#endif
    _interm_inner_points.copySizeFromDevice( _stream, EdgeListWait );

#ifndef NDEBUG
    cerr << "Debug voting (without separable compilation)"
         << " # inner points: " << _interm_inner_points.host.size
         << " # voters      : " << _voters.host.size << endl;
#endif

    /* Add number of voters to chosen inner points, and
     * add average flow length to chosen inner points.
     */
    dim3 block( 32, 1, 1 );
    dim3 grid ( grid_divide( _interm_inner_points.host.size, 32 ), 1, 1 );

    vote::eval_chosen
        <<<grid,block,0,_stream>>>
        ( _meta,
          _voters.dev,
          _interm_inner_points.dev );
    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_EVAL

} // namespace popart

