#include "onoff.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart {
namespace vote {

__device__ inline
TriplePoint* find_neigh( const int2&              neigh,
                         cv::cuda::PtrStepSz32s   edgepoint_index_table,
                         DevEdgeList<TriplePoint> chained_edgecoords )
{
    if( neigh.x != 0 || neigh.y != 0 ) {
        int idx = edgepoint_index_table.ptr(neigh.y)[neigh.x];
        if( idx != 0 ) {
            assert( idx > 0 );
            assert( idx < chained_edgecoords.Size() );
            TriplePoint* neighbour = &chained_edgecoords.ptr[idx];
#ifndef NDEBUG
            debug_inner_test_consistency( "B", idx, neighbour, edgepoint_index_table, chained_edgecoords );

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

__device__ inline
TriplePoint* find_befor( const TriplePoint*       p,
                         cv::cuda::PtrStepSz32s   edgepoint_index_table,
                         DevEdgeList<TriplePoint> chained_edgecoords )
{
    assert( p );
    return find_neigh( p->descending.befor,
                       edgepoint_index_table,
                       chained_edgecoords );
}


__device__ inline
TriplePoint* find_after( const TriplePoint*             p,
                               cv::cuda::PtrStepSz32s   edgepoint_index_table,
                               DevEdgeList<TriplePoint> chained_edgecoords )
{
    assert( p );
    return find_neigh( p->descending.after,
                       edgepoint_index_table,
                       chained_edgecoords );
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
inline float cl_distance( const TriplePoint* l, const TriplePoint* r )
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
const TriplePoint* cl_inner(
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
    debug_inner_test_consistency( "A", offset, p, edgepoint_index_table, chained_edgecoords );
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

    float lastDist = vote::cl_distance( p, current ); // hypotf is CUDA float intrinsic for sqrt(pow2+pow2)
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

        dist = vote::cl_distance( target, current );
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

        dist = vote::cl_distance( target, current );
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
                     const cv::cuda::PtrStepSz32s edgepoint_index_table, // input
                     const size_t                 numCrowns,
                     const float                  ratioVoting )
{
    const TriplePoint* chosen =
        cl_inner( chained_edgecoords,    // input
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
        write_index = atomicAdd( seed_indices.getSizePtr(), (int)ct );
    }
    write_index = __shfl( write_index, 0 );
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    if( chosen ) {
        if( seed_indices.Size() < EDGE_POINT_MAX ) {
            idx = edgepoint_index_table.ptr(chosen->coord.y)[chosen->coord.x];
            seed_indices.ptr[write_index] = idx;
        }
    }
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION
__global__
void applyVoteConstructLine_dp(
    DevEdgeList<int>         seedIndices,         // output
    DevEdgeList<TriplePoint> chainedEdgeCoords,   // ?
    cv::cuda::PtrStepSz32s   edgepointIndexTable, // ?
    const size_t             param_numCrowns,     // input param
    const float              param_ratioVoting )  // input param
{
    int listsize = chainedEdgeCoords.getSize();

    if( listsize == 0 ) return;

    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( listsize, 32 ), 1, 1 );

    seedIndices.setSize( 0 );

    vote::construct_line
        <<<grid,block>>>
        ( seedIndices,         // output
          chainedEdgeCoords,   // ?
          edgepointIndexTable, // ?
          param_numCrowns,     // input
          param_ratioVoting ); // input
}

__host__
bool Frame::applyVoteConstructLine( const cctag::Parameters& params )
{
    applyVoteConstructLine_dp
        <<<1,1,0,_stream>>>
        ( _vote._seed_indices.dev,        // output
          _vote._chained_edgecoords.dev,  // ?
          _vote._d_edgepoint_index_table, // ?
          params._nCrowns,                // input param
          params._ratioVoting );          // input param
    POP_CHK_CALL_IFSYNC;
    return true;
}

#else // not USE_SEPARABLE_COMPILATION
__host__
bool Frame::applyVoteConstructLine( const cctag::Parameters& params )
{
    // Note: right here, Dynamic Parallelism would avoid blocking.
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._chained_edgecoords.host.size,
                                   _vote._chained_edgecoords.dev.getSizePtr(),
                                   sizeof(int), _stream );
    POP_CUDA_SYNC( _stream );

    int listsize = _vote._chained_edgecoords.host.size;

    if( listsize == 0 ) {
        return false;
    }

    dim3 block( 32, 1, 1 );
    dim3 grid ( grid_divide( listsize, 32 ), 1, 1 );

    POP_CUDA_SET0_ASYNC( _vote._seed_indices.dev.getSizePtr(), _stream );

    vote::construct_line
        <<<grid,block,0,_stream>>>
        ( _vote._seed_indices.dev,        // output
          _vote._chained_edgecoords.dev,  // input
          _vote._d_edgepoint_index_table, // input
          params._nCrowns,                // input param
          params._ratioVoting );          // input param
    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION

} // namespace popart

