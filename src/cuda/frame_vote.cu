#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <thrust/system/cuda/detail/cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"
#include "cctag/talk.hpp" // for DO_TALK macro

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

namespace vote
{

__device__
inline
TriplePoint* find_neigh( const int2&              neigh,
                         cv::cuda::PtrStepSz32s   edgepoint_index_table,
                         DevEdgeList<TriplePoint> chained_edgecoords )
{
    if( neigh.x != 0 || neigh.y != 0 ) {
        int idx = edgepoint_index_table.ptr(neigh.y)[neigh.x];
        if( idx != 0 ) {
            assert( idx >= 0 && idx < chained_edgecoords.Size() );
            TriplePoint* neighbour = &chained_edgecoords.ptr[idx];
#ifndef NDEBUG
            debug_inner_test_consistency( idx, neighbour, edgepoint_index_table, chained_edgecoords );

            if( neigh.x != neighbour->coord.x || neigh.y != neighbour->coord.y ) {
                printf("Intended coordinate is (%d,%d) at index %d, looked up coord is (%d,%d)\n",
                       neigh.x, neigh.y,
                       idx,
                       neighbour->coord.x, neighbour->coord.y );
            }
#endif // NDEBUG
            return neighbour;
        }
    }
    return 0;
}

__device__
inline
TriplePoint* find_befor( const TriplePoint*       p,
                         cv::cuda::PtrStepSz32s   edgepoint_index_table,
                         DevEdgeList<TriplePoint> chained_edgecoords )
{
    return find_neigh( p->descending.befor, edgepoint_index_table, chained_edgecoords );
}


__device__
inline
TriplePoint* find_after( const TriplePoint*             p,
                               cv::cuda::PtrStepSz32s   edgepoint_index_table,
                               DevEdgeList<TriplePoint> chained_edgecoords )
{
    return find_neigh( p->descending.after, edgepoint_index_table, chained_edgecoords );
}

__device__
float inner_prod( const TriplePoint* l,
                  const TriplePoint* r )
{
    assert( l );
    assert( r );
    const int16_t l_dx = l->d.x;
    const int16_t l_dy = l->d.y;
    const int16_t r_dx = r->d.x;
    const int16_t r_dy = r->d.y;
    assert( l_dx != 0 || l_dy != 0 );
    assert( r_dx != 0 || r_dy != 0 );
    const float ret  = (float)l_dx * (float)r_dx
                     + (float)l_dy * (float)r_dy;
    return ret;

    // Point2dN l ( l_dx, l_dy );
    // Point2dN r ( r_dx, r_dy );
    // float return = -inner_prod(subrange(l, 0, 2), subrange(r, 0, 2));
}

__device__
inline float distance( const TriplePoint* l, const TriplePoint* r )
{
    return hypotf( l->coord.x - r->coord.x, l->coord.y - r->coord.y );
}

/* Brief: Voting procedure. For every edge point, construct the 1st order approximation 
 * of the field line passing through it, which consists in a polygonal line whose
 * extremities are two edge points.
 * Input:
 * points: set of edge points to be processed, i.e. considered as the 1st extremity
 * of the constructed field line passing through it.
 * seeds: edge points having received enough votes to be considered as a seed, i.e.
 * as an edge point belonging on an inner elliptical arc of a cctag.
 * edgesMap: map of all the edge points
 * winners: map associating all seeds to their voters
 * cannyGradX: X derivative of the gray image
 * cannyGradY: Y derivative of the gray image
 */
__device__
const TriplePoint* construct_line_inner(
    DevEdgeList<TriplePoint>       chained_edgecoords,
    const cv::cuda::PtrStepSz32s   edgepoint_index_table,
    const size_t                   numCrowns,
    const float                    ratioVoting )
{
    int offset = threadIdx.x + blockIdx.x * 32;
    if( offset >= chained_edgecoords.Size() ) {
        return 0;
    }
    if( offset == 0 ) {
        /* special case: offset 0 is intentionally empty */
        return 0;
    }

    TriplePoint* const p = &chained_edgecoords.ptr[offset];

    if( p == 0 ) return 0;

#ifndef NDEBUG
    p->debug_init( );
    debug_inner_test_consistency( offset, p, edgepoint_index_table, chained_edgecoords );
    p->debug_add( p->coord );
#endif // NDEBUG

    float dist; // scalar to compute the distance ratio

    TriplePoint* current = vote::find_befor( p, edgepoint_index_table, chained_edgecoords );
    // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
    if( not current ) {
        return 0;
    }
#ifndef NDEBUG
    p->debug_add( current->coord );
#endif

    // To save all sub-segments length
    int       vDistSize = 0;
#ifndef NDEBUG
    const int vDistMax  = numCrowns * 2 - 1;
#endif // NDEBUG
    float     vDist[RESERVE_MEM_MAX_CROWNS * 2 - 1];
    int flagDist = 1;

    // Length of the reconstructed field line approximation between the two
    // extremities.
    float totalDistance = 0.0;

    // compute difference in subsequent gradients orientation
    float cosDiffTheta = -vote::inner_prod( p, current );
    if( cosDiffTheta < 0.0 ) {
        return 0;
    }

    float lastDist = vote::distance( p, current ); // hypotf is CUDA float intrinsic for sqrt(pow2+pow2)
    vDist[vDistSize++] = lastDist;
    assert( vDistSize <= vDistMax );
        
    // Add the sub-segment length to the total distance.
    totalDistance += lastDist;

    TriplePoint* chosen = 0;

    // Iterate over all crowns
    for( int i=1; i < numCrowns; ++i ) {
        chosen = 0;

        // First in the gradient direction
        TriplePoint* target = vote::find_after( current,
                                                edgepoint_index_table,
                                                chained_edgecoords );
        // No edge point was found in that direction
        if( not target ) {
            return 0;
        }
#ifndef NDEBUG
        p->debug_add( target->coord );
#endif

        // Check the difference of two consecutive angles
        cosDiffTheta = -vote::inner_prod( target, current );
        if( cosDiffTheta < 0.0 ) {
            return 0;
        }

        dist = vote::distance( target, current );
        vDist[vDistSize++] = dist;
        assert( vDistSize <= vDistMax );
        totalDistance += dist;

        // Check the distance ratio
        if( vDistSize > 1 ) {
            for( int iDist = 0; iDist < vDistSize; ++iDist ) {
                for (int jDist = iDist + 1; jDist < vDistSize; ++jDist) {
                    flagDist = (vDist[iDist] <= vDist[jDist] * ratioVoting) && (vDist[jDist] <= vDist[iDist] * ratioVoting) && flagDist;
                }
            }
        }

        if( not flagDist ) {
            return 0;
        }

        // lastDist = dist;
        current = target;
        // Second in the opposite gradient direction
        // target = vote::find_befor( current, d_next_edge_befor, chained_edgecoord_list );
        target = vote::find_befor( current,
                                   edgepoint_index_table,
                                   chained_edgecoords );
        if( not target ) {
            return 0;
        }

        cosDiffTheta = -vote::inner_prod( target, current );
        if( cosDiffTheta < 0.0 ) {
            return 0;
        }

        dist = vote::distance( target, current );
        vDist[vDistSize++] = dist;
        assert( vDistSize <= vDistMax );
        totalDistance += dist;

        for( int iDist = 0; iDist < vDistSize; ++iDist ) {
            for (int jDist = iDist + 1; jDist < vDistSize; ++jDist) {
                flagDist = (vDist[iDist] <= vDist[jDist] * ratioVoting) && (vDist[jDist] <= vDist[iDist] * ratioVoting) && flagDist;
            }
        }

        if( not flagDist ) {
            return 0;
        }

        // lastDist = dist;
        current = target;
        chosen = current;

        if( !current ) {
            return 0;
        }
#ifndef NDEBUG
        p->debug_add( current->coord );
#endif
    }
#ifndef NDEBUG
    p->debug_commit( );
#endif

    /* The overhead of competing updates in the chosen points
     * would be huge.
     * But every point chooses at most one chosen, so we can
     * keep the important data in the choosers for now, and
     * update the chosen in a new kernel.
     */
    p->my_vote            = edgepoint_index_table.ptr(chosen->coord.y)[chosen->coord.x];
    p->chosen_flow_length = totalDistance;

    return chosen;
}

__global__
void construct_line( DevEdgeList<int>             seed_indices,       // output
                     DevEdgeList<TriplePoint>     chained_edgecoords, // input/output
                     const int                    edge_index_max,     // input
                     const cv::cuda::PtrStepSz32s edgepoint_index_table, // input
                     const size_t                 numCrowns,
                     const float                  ratioVoting )
{
    const TriplePoint* chosen =
        construct_line_inner( chained_edgecoords,    // input
                              edgepoint_index_table, // input
                              numCrowns,
                              ratioVoting );

    if( chosen && chosen->coord.x == 0 && chosen->coord.y == 0 ) chosen = 0;

    int idx = 0;
    uint32_t mask   = __ballot( chosen != 0 );
    uint32_t ct     = __popc( mask );
    if( ct == 0 ) return;

    uint32_t write_index;
    if( threadIdx.x == 0 ) {
        write_index = atomicAdd( seed_indices.size, (int)ct );
    }
    write_index = __shfl( write_index, 0 );
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    if( chosen ) {
        if( seed_indices.Size() < edge_index_max ) {
            idx = edgepoint_index_table.ptr(chosen->coord.y)[chosen->coord.x];
            seed_indices.ptr[write_index] = idx;
        }
    }
}

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

} // namespace vote

/* For all chosen inner points, compute the average flow length and the
 * number of voters, and store in the TriplePoint structure of the chosen
 * inner point.
 *
 * chained_edgecoords is the list of all edges with their chaining info.
 * seed_indices is a list of indices into that list, containing the sorted,
 * unique indices of chosen inner points.
 */
__global__
void vote_eval_chosen( DevEdgeList<TriplePoint> chained_edgecoords, // input-output
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

struct NumVotersIsGreaterEqual
{
    DevEdgeList<TriplePoint> _array;
    int                      _compare;

    CUB_RUNTIME_FUNCTION
    __host__ __device__
    __forceinline__
    NumVotersIsGreaterEqual( int compare, DevEdgeList<TriplePoint> _d_array )
        : _compare(compare)
        , _array( _d_array )
    {}

    // CUB_RUNTIME_FUNCTION
    __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_array.ptr[a]._winnerSize >= _compare);
    }
};

__host__
bool Voting::constructLine( const cctag::Parameters&     params,
                            cudaStream_t                 stream )
{
    // Note: right here, Dynamic Parallelism would avoid blocking.
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_chained_edgecoords.host.size,
                                   _chained_edgecoords.dev.size,
                                   sizeof(int), stream );
    POP_CUDA_SYNC( stream );

    int listsize = _chained_edgecoords.host.size;

    if( listsize == 0 ) {
        return false;
    }

    dim3 block;
    dim3 grid;

    block.x = 32;
    block.y = 1;
    block.z = 1;
    grid.x  = listsize / 32 + ( listsize % 32 != 0 ? 1 : 0 );
    grid.y  = 1;
    grid.z  = 1;

    POP_CUDA_SET0_ASYNC( _seed_indices.dev.size, stream );

    vote::construct_line
        <<<grid,block,0,stream>>>
        ( _seed_indices.dev,        // output
          _chained_edgecoords.dev,  // input
          params._maxEdges,         // input
          _d_edgepoint_index_table, // input
          params._nCrowns,          // input
          params._ratioVoting );    // input
    POP_CHK_CALL_IFSYNC;

    return true;
}

__host__
void Frame::applyVote( const cctag::Parameters& params )
{
    bool success;

    success = _vote.constructLine( params,
                                   _stream );

    if( not success ) {
        _vote._seed_indices.host.size       = 0;
        _vote._chained_edgecoords.host.size = 0;
        return;
    }

    /* For every chosen, compute the average flow size from all
     * of its voters, and count the number of its voters.
     */
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._seed_indices.host.size, _vote._seed_indices.dev.size, sizeof(int), _stream );
    POP_CUDA_SYNC( _stream );

#ifndef NDEBUG
#ifdef  DEBUG_RETURN_AFTER_CONSTRUCT_LINE
    {
        /* _vote._seed_indices contains now the indices of all TriplePoints that
         * have received at least one vote.
         * The array has lots of redundant entries. It is not sorted, and the
         * number of voters has not been counted, and it has not been filtered
         * by length or voters count.
         */
        return;
    }
#endif //  DEBUG_RETURN_AFTER_CONSTRUCT_LINE
#endif // NDEBUG

    if( _vote._seed_indices.host.size > 0 ) {
        /* Note: we use the intermediate picture plane, _d_intermediate, as assist
         *       buffer for CUB algorithms. It is extremely likely that this plane
         *       is large enough in all cases. If there are any problems, call
         *       the function with assist_buffer=0, and the function will return
         *       the required size in assist_buffer_sz (call by reference).
         */
        void*  assist_buffer = (void*)_d_intermediate.data;
        size_t assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        cub::DoubleBuffer<int> d_keys( _vote._seed_indices.dev.ptr,
                                       _vote._seed_indices_2.dev.ptr );

        /* After SortKeys, both buffers in d_keys have been altered.
         * The final result is stored in d_keys.d_buffers[d_keys.selector].
         * The other buffer is invalid.
         */
        cub::DeviceRadixSort::SortKeys( assist_buffer,
                                        assist_buffer_sz,
                                        d_keys,
                                        _vote._seed_indices.host.size,
                                        0,             // begin_bit
                                        sizeof(int)*8, // end_bit
                                        _stream );
        POP_CHK_CALL_IFSYNC;

        if( d_keys.d_buffers[d_keys.selector] == _vote._seed_indices_2.dev.ptr ) {
            std::swap( _vote._seed_indices.dev.ptr,   _vote._seed_indices_2.dev.ptr );
            std::swap( _vote._seed_indices.dev.size,  _vote._seed_indices_2.dev.size );
        }

        // safety: SortKeys is allowed to alter assist_buffer_sz
        assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        /* Unique ensure that we check every "chosen" point only once.
         * Output is in _vote._seed_indices_2.dev
         */
        cub::DeviceSelect::Unique( assist_buffer,
                                   assist_buffer_sz,
                                   _vote._seed_indices.dev.ptr,     // input
                                   _vote._seed_indices_2.dev.ptr,   // output
                                   _vote._seed_indices_2.dev.size,  // output
                                   _vote._seed_indices.host.size,   // input (unchanged in sort)
                                   _stream );
        POP_CHK_CALL_IFSYNC;

        /* Without Dynamic Parallelism, we must block here to retrieve the
         * value d_num_selected_out from the device before the voting
         * step.
         */
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._seed_indices_2.host.size,
                                       _vote._seed_indices_2.dev.size,
                                       sizeof(int), _stream );
        POP_CUDA_SYNC( _stream );

        /* Add number of voters to chosen inner points, and
         * add average flow length to chosen inner points.
         */
        dim3 block;
        dim3 grid;

        block.x = 32;
        block.y = 1;
        block.z = 1;
        grid.x  = grid_divide( _vote._seed_indices_2.host.size, 32 );
        grid.y  = 1;
        grid.z  = 1;

        vote_eval_chosen
            <<<grid,block,0,_stream>>>
            ( _vote._chained_edgecoords.dev,
              _vote._seed_indices_2.dev );
        POP_CHK_CALL_IFSYNC;

#ifdef EDGE_LINKING_HOST_SIDE
        /* After vote_eval_chosen, _chained_edgecoords is no longer changed
         * we can copy it to the host for edge linking
         */
        _vote._chained_edgecoords.copySizeFromDevice( _stream );
        POP_CUDA_SYNC( _stream );
        _vote._chained_edgecoords.copyDataFromDevice( _vote._chained_edgecoords.host.size,
                                                      _stream );
        POP_CHK_CALL_IFSYNC;
#endif // EDGE_LINKING_HOST_SIDE

        // safety: SortKeys is allowed to alter assist_buffer_sz
        assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        /* Filter all chosen inner points that have fewer
         * voters than required by Parameters.
         */
        NumVotersIsGreaterEqual select_op( params._minVotesToSelectCandidate,
                                           _vote._chained_edgecoords.dev );
        cub::DeviceSelect::If( assist_buffer,
                               assist_buffer_sz,
                               _vote._seed_indices_2.dev.ptr,
                               _vote._seed_indices.dev.ptr,
                               _vote._seed_indices.dev.size,
                               _vote._seed_indices_2.host.size,
                               select_op,
                               _stream );
        POP_CHK_CALL_IFSYNC;

        _vote._seed_indices.copySizeFromDevice( _stream );
        POP_CUDA_SYNC( _stream );
#ifdef EDGE_LINKING_HOST_SIDE
        if( _vote._seed_indices.host.size != 0 ) {
            _vote._seed_indices.copyDataFromDevice( _vote._seed_indices.host.size, _stream );
        }
#endif // EDGE_LINKING_HOST_SIDE
    } else {
        _vote._chained_edgecoords.host.size = 0;
    }
}

} // namespace popart

