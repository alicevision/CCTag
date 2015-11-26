#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// #include <thrust/system/cuda/detail/cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

namespace vote
{

__device__ inline
int count_winners( const int                       chosen_edge_index,
                   TriplePoint*                    chosen_edge,
                   const DevEdgeList<TriplePoint>& array )
{
    int   winner_size = 0;
    float flow_length = 0.0f;

    /* This loop looks dangerous, but it is actually faster than
     * a manually partially unrolled loop.
     */
    const int voter_list_size = array.Size();
    for( int i=0; i<voter_list_size; i++ )
    // for( int i=0; i<chained_edgecoords.Size(); i++ )
    {
        if( array.ptr[i].my_vote == chosen_edge_index ) {
            winner_size += 1;
            flow_length += array.ptr[i].chosen_flow_length;
        }
    }
    chosen_edge->_winnerSize = winner_size;
    chosen_edge->_flowLength = flow_length / winner_size;
    return winner_size;
}

/* For all chosen inner points, compute the average flow length and the
 * number of voters, and store in the TriplePoint structure of the chosen
 * inner point.
 *
 * chained_edgecoords is the list of all edges with their chaining info.
 * seed_indices is a list of indices into that list, containing the sorted,
 * unique indices of chosen inner points.
 */
__global__
void eval_chosen( DevEdgeList<TriplePoint> chained_edgecoords, // input-output
                  DevEdgeList<int>         seed_indices        // input
                )
{
    uint32_t offset = threadIdx.x + blockIdx.x * 32;
    if( offset >= seed_indices.Size() ) {
        return;
    }

    const int    chosen_edge_index = seed_indices.ptr[offset];
    TriplePoint* chosen_edge = &chained_edgecoords.ptr[chosen_edge_index];
    vote::count_winners( chosen_edge_index, chosen_edge, chained_edgecoords );
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION_IN_GRADDESC
// this is called in frame_desc.cu by descent::dp_caller
#else // USE_SEPARABLE_COMPILATION_IN_GRADDESC
#endif // USE_SEPARABLE_COMPILATION_IN_GRADDESC

#ifdef USE_SEPARABLE_COMPILATION_IN_GRADDESC
__host__
void Frame::applyVote( const cctag::Parameters& )
{
    // everything that was in here is called frame_desc.cu by
    // descent::dp_caller when USE_SEPARABLE_COMPILATION is
    // used
}
#else // not USE_SEPARABLE_COMPILATION_IN_GRADDESC
__host__
void Frame::applyVote( const cctag::Parameters& params )
{
    bool success;
    cudaError_t err;

    success = applyVoteConstructLine( params );

    if( not success ) {
        _vote._seed_indices.host.size       = 0;
        _vote._chained_edgecoords.host.size = 0;
        return;
    }

    /* For every chosen, compute the average flow size from all
     * of its voters, and count the number of its voters.
     */
    success = applyVoteSortUniqNoDP( params );

    if( success ) {
        void*  assist_buffer = (void*)_d_intermediate.data;
        size_t assist_buffer_sz;

        /* Without Dynamic Parallelism, we must block here to retrieve the
         * value d_num_selected_out from the device before the voting
         * step.
         */
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._seed_indices_2.host.size,
                                       _vote._seed_indices_2.dev.getSizePtr(),
                                       sizeof(int), _stream );
        POP_CUDA_SYNC( _stream );

        /* Add number of voters to chosen inner points, and
         * add average flow length to chosen inner points.
         */
        dim3 block( 32, 1, 1 );
        dim3 grid ( grid_divide( _vote._seed_indices_2.host.size, 32 ), 1, 1 );

        vote::eval_chosen
            <<<grid,block,0,_stream>>>
            ( _vote._chained_edgecoords.dev,
              _vote._seed_indices_2.dev );
        POP_CHK_CALL_IFSYNC;

        NumVotersIsGreaterEqual select_op( params._minVotesToSelectCandidate,
                                           _vote._chained_edgecoords.dev );
#ifdef CUB_INIT_CALLS
	    assist_buffer_sz  = 0;
        err = cub::DeviceSelect::If( 0,
                                     assist_buffer_sz,
                                     _vote._seed_indices_2.dev.ptr,
                                     _vote._seed_indices.dev.ptr,
                                     _vote._seed_indices.dev.getSizePtr(),
                                     _vote._seed_indices_2.host.size,
                                     select_op,
                                     _stream,
                                     DEBUG_CUB_FUNCTIONS );

	if( err != cudaSuccess ) {
	    std::cerr << "cub::DeviceSelect::If init step failed. Crashing." << std::endl;
	    std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
	    exit(-1);
	}
	if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
            std::cerr << "cub::DeviceSelect::If requires too much intermediate memory. Crashing." << std::endl;
	    exit( -1 );
	}
#else
	// THIS CODE WORKED BEFORE
        // safety: SortKeys is allowed to alter assist_buffer_sz
        assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;
#endif

        /* Filter all chosen inner points that have fewer
         * voters than required by Parameters.
         */

        err = cub::DeviceSelect::If( assist_buffer,
                                     assist_buffer_sz,
                                     _vote._seed_indices_2.dev.ptr,
                                     _vote._seed_indices.dev.ptr,
                                     _vote._seed_indices.dev.getSizePtr(),
                                     _vote._seed_indices_2.host.size,
                                     select_op,
                                     _stream,
                                     DEBUG_CUB_FUNCTIONS );
        POP_CHK_CALL_IFSYNC;
        POP_CUDA_FATAL_TEST( err, "CUB DeviceSelect::If failed" );

        _vote._seed_indices.copySizeFromDevice( _stream );
        POP_CUDA_SYNC( _stream );
#ifdef EDGE_LINKING_HOST_SIDE
        /* After vote_eval_chosen, _chained_edgecoords is no longer changed
         * we can copy it to the host for edge linking
         */
        _vote._chained_edgecoords.copySizeFromDevice( _stream );
        POP_CUDA_SYNC( _stream );
        _vote._chained_edgecoords.copyDataFromDevice( _vote._chained_edgecoords.host.size,
                                                      _stream );
        POP_CHK_CALL_IFSYNC;

        if( _vote._seed_indices.host.size != 0 ) {
            _vote._seed_indices.copyDataFromDevice( _vote._seed_indices.host.size, _stream );
        }
#endif // EDGE_LINKING_HOST_SIDE
    } else {
        _vote._chained_edgecoords.host.size = 0;
    }
}
#endif // not USE_SEPARABLE_COMPILATION_IN_GRADDESC

} // namespace popart

