#include "onoff.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "frameparam.h"
#include "assist.h"

using namespace std;

namespace popart {
namespace vote {

__device__ inline
CudaEdgePoint* find_neigh( const int                  neigh,
                           DevEdgeList<CudaEdgePoint> d_edgepoints )
{
    if( neigh >= 0 ) {
        CudaEdgePoint* neighbour = &d_edgepoints.ptr[neigh];
        return neighbour;
    }
    return 0;
}

__device__ inline
CudaEdgePoint* find_befor( const CudaEdgePoint*       p,
                           DevEdgeList<CudaEdgePoint> d_edgepoints )
{
    assert( p );
    return find_neigh( p->_dev_befor, d_edgepoints );
}


__device__ inline
CudaEdgePoint* find_after( const CudaEdgePoint*       p,
                           DevEdgeList<CudaEdgePoint> d_edgepoints )
{
    assert( p );
    return find_neigh( p->_dev_after, d_edgepoints );
}

__device__
float inner_prod( const CudaEdgePoint* l,
                  const CudaEdgePoint* r )
{
    assert( l );
    assert( r );
    const int16_t l_dx = l->_grad.x;
    const int16_t l_dy = l->_grad.y;
    const int16_t r_dx = r->_grad.x;
    const int16_t r_dy = r->_grad.y;
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
inline float cl_distance( const CudaEdgePoint* l, const CudaEdgePoint* r )
{
    return hypotf( l->_coord.x - r->_coord.x, l->_coord.y - r->_coord.y );
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
const CudaEdgePoint* cl_inner(
    FrameMetaPtr&              meta,
    DevEdgeList<CudaEdgePoint> d_edgepoints, // input
    DevEdgeList<int>           voters,       // input
    const int                  voter_offset, // input
    float&                     flow_length )
{
    if( voter_offset >= meta.list_size_voters() ) {
        return 0;
    }

    const int chooser_offset = voters.ptr[voter_offset];

    if( chooser_offset < 0 ) {
        return 0;
    }

    CudaEdgePoint* const p = &d_edgepoints.ptr[chooser_offset];

    if( p == 0 ) return 0;

    float dist; // scalar to compute the distance ratio

    CudaEdgePoint* current = vote::find_befor( p, d_edgepoints );
    // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
    if( not current ) {
        return 0;
    }

    // To save all sub-segments length
    int       vDistSize = 0;
#ifndef NDEBUG
    const int vDistMax  = tagParam.nCrowns * 2 - 1;
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

    float lastDist = vote::cl_distance( p, current );
    vDist[vDistSize++] = lastDist;
    assert( vDistSize <= vDistMax );
        
    // Add the sub-segment length to the total distance.
    totalDistance += lastDist;

    CudaEdgePoint* chosen = 0;

    // Iterate over all crowns
    for( int i=1; i < tagParam.nCrowns; ++i ) {
        chosen = 0;

        // First in the gradient direction
        CudaEdgePoint* target = vote::find_after( current, d_edgepoints );
        // No edge point was found in that direction
        if( not target ) {
            return 0;
        }

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
                    flagDist = (vDist[iDist] <= vDist[jDist] * tagParam.ratioVoting) && (vDist[jDist] <= vDist[iDist] * tagParam.ratioVoting) && flagDist;
                }
            }
        }

        if( not flagDist ) {
            return 0;
        }

        // lastDist = dist;
        current = target;
        // Second in the opposite gradient direction
        // target = vote::find_befor( current, d_next_edge_befor, voters );
        target = vote::find_befor( current, d_edgepoints );
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
                flagDist = (vDist[iDist] <= vDist[jDist] * tagParam.ratioVoting) && (vDist[jDist] <= vDist[iDist] * tagParam.ratioVoting) && flagDist;
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
    }

    /* The overhead of competing updates in the chosen points
     * would be huge.
     * But every point chooses at most one chosen, so we can
     * keep the important data in the choosers for now, and
     * update the chosen in a new kernel.
     */
    // p->my_vote            = d_edgepoint_map.ptr(chosen->coord.y)[chosen->coord.x];
    // p->vote_weight = totalDistance;
    // const int index_of_chosen = d_edgepoints.ptr(chosen->coord.y)[chosen->coord.x];

    // voting_for.ptr[offset]     = index_of_chosen;
    flow_length = totalDistance;

    return chosen;
}

__global__
void construct_line( FrameMetaPtr               meta,
                     DevEdgeList<CudaEdgePoint> d_edgepoints,       // input-output
                     cv::cuda::PtrStepSz32s     d_edgepoint_map,    // input
                     DevEdgeList<int>           voters,             // input
                     DevEdgeList<int>           inner_points,       // output
                     DevEdgeList<float>         vote_weight, // output
                     DevEdgeList<int>           voting_for )        // output
{
    const int voter_offset   = threadIdx.x + blockIdx.x * 32;

    float flow_length = -1.0f;

    const CudaEdgePoint* chosen =
        cl_inner( meta,
                  d_edgepoints,         // input
                  voters,               // input
                  voter_offset,         // input
                  flow_length ); // output

    const int chosen_offset  = chosen ? d_edgepoint_map.ptr(chosen->_coord.y)[chosen->_coord.x]
                                      : -1;
    voting_for .ptr[voter_offset] = chosen_offset;
    vote_weight.ptr[voter_offset] = flow_length;

    int idx = 0;
    uint32_t mask   = __ballot( chosen != 0 );
    uint32_t ct     = __popc( mask );
    if( ct == 0 ) return;

    uint32_t write_index;
    if( threadIdx.x == 0 ) {
        write_index = atomicAdd( &meta.list_size_inner_points(), (int)ct );
    }
    write_index = __shfl( write_index, 0 );
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    if( chosen ) {
        if( meta.list_size_inner_points() < EDGE_POINT_MAX ) {
            idx = d_edgepoint_map.ptr(chosen->_coord.y)[chosen->_coord.x];
            inner_points.ptr[write_index] = idx;
        }
    }
}

} // namespace vote

#ifdef USE_SEPARABLE_COMPILATION_FOR_VOTE_LINE
__global__
void dp_call_construct_line(
    FrameMetaPtr               meta,
    DevEdgeList<CudaEdgePoint> d_edgepoints,         // input-output
    cv::cuda::PtrStepSz32s     d_edgepoint_map,      // input
    DevEdgeList<int>           voters,               // input
    DevEdgeList<int>           inner_points,         // output
    DevEdgeList<float>         vote_weight,   // output
    DevEdgeList<int>           voting_for )          // output
{
    meta.list_size_inner_points() = 0;

    int num_voters = meta.list_size_voters();

    if( num_voters == 0 ) return;

    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( num_voters, 32 ), 1, 1 );

    vote::construct_line
        <<<grid,block>>>
        ( meta,
          d_edgepoints,
          d_edgepoint_map,
          voters,
          inner_points,           // output
          vote_weight,
          voting_for );
}

__host__
bool Frame::applyVoteConstructLine( )
{
    _inner_points.host.size = 0;

    dp_call_construct_line
        <<<1,1,0,_stream>>>
        ( _meta,
          _edgepoints.dev,   // input
          _d_edgepoint_map,  // input 2D->edge index mapping
          _voters.dev,       // input
          _inner_points.dev, // output
          _vote_weight.dev,  // output
          _voting_for.dev ); // output
    POP_CHK_CALL_IFSYNC;

#if 1
    _inner_points.copySizeFromDevice( _stream, EdgeListWait );
    cudaDeviceSynchronize();
    _inner_points.copyDataFromDeviceSync( );
    std::vector<int> out;
    _inner_points.debug_out( EDGE_POINT_MAX, out, EdgeListFilterAny );
#endif

    return true;
}

#else // not USE_SEPARABLE_COMPILATION_FOR_VOTE_LINE
__host__
bool Frame::applyVoteConstructLine( )
{
    _voters.copySizeFromDevice( _stream, EdgeListWait );

    if( _voters.host.size == 0 ) {
        return false;
    }

    _inner_points.host.size = 0;
    _inner_points.copySizeToDevice( _stream, EdgeListCont );

    dim3 block( 32, 1, 1 );
    dim3 grid ( grid_divide( _voters.host.size, 32 ), 1, 1 );

    vote::construct_line
        <<<grid,block,0,_stream>>>
        ( _meta,
          _edgepoints.dev,   // input-output
          _d_edgepoint_map,  // input 2D->edge index mapping
          _voters.dev,       // input
          _inner_points.dev, // output
          _vote_weight.dev,  // output
          _voting_for.dev ); // output

    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_VOTE_LINE

} // namespace popart

