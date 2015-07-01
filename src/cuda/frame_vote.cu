#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "assist.h"

using namespace std;

#define MAX_CROWNS  5

namespace popart
{

namespace vote
{

#ifndef NDEBUG
__device__
void inner_test_consistency( int p_idx, const TriplePoint* p, cv::cuda::PtrStepSz32s point_map, const TriplePoint* point_list, uint32_t point_list_len )
{
    if( p == 0 ) {
        printf("Impossible bug, initialized from memory address\n");
        assert( 0 );
    }

    if( outOfBounds( p->coord, point_map ) ) {
        printf("Index (%d,%d) does not fit into coord lookup tables\n", p->coord.x, p->coord.y );
        assert( 0 );
    }

    int idx = point_map.ptr(p->coord.y)[p->coord.x];
    if( idx < 0 || idx >= point_list_len ) {
        printf("Looked up index (coord) is out of bounds\n");
        assert( 0 );
    }

    if( idx != p_idx ) {
        printf("Looked up index %d is not identical to input index %d\n", idx, p_idx);
        assert( 0 );
    }

    if( outOfBounds( p->befor, point_map ) ) {
        printf("Before coordinations (%d,%d) do not fit into lookup tables\n", p->befor.x, p->befor.y );
        assert( 0 );
    }

    if( outOfBounds( p->after, point_map ) ) {
        printf("After coordinations (%d,%d) do not fit into lookup tables\n", p->after.x, p->after.y );
        assert( 0 );
    }
}
#endif // NDEBUG

__device__
inline
const TriplePoint* find_befor( const TriplePoint*     p,
                               cv::cuda::PtrStepSz32s d_next_edge_coord,
                               const TriplePoint*     d_edgelist_2,
                               int                    d_edgelist_2_sz )
{
    const TriplePoint* before = 0;
    int                idx    = 0;
    if( p->befor.x != 0 && p->befor.y != 0 ) {
        idx = d_next_edge_coord.ptr(p->befor.y)[p->befor.x];
        if( idx != 0 ) {
            assert( idx >= 0 && idx < d_edgelist_2_sz );
            before = &d_edgelist_2[idx];
#ifndef NDEBUG
            inner_test_consistency( idx, before, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );

            if( p->befor.x != before->coord.x || p->befor.y != before->coord.y ) {
                printf("Intended coordinate is (%d,%d) at index %d, looked up coord is (%d,%d)\n",
                       p->befor.x, p->befor.y,
                       idx,
                       before->coord.x, before->coord.y );
            }
#endif // NDEBUG
        }
    }
    return before;
}


__device__
inline
const TriplePoint* find_after( const TriplePoint*     p,
                               cv::cuda::PtrStepSz32s d_next_edge_coord,
                               const TriplePoint*     d_edgelist_2,
                               int                    d_edgelist_2_sz )
{
    const TriplePoint* after = 0;
    int                idx    = 0;
    if( p->after.x != 0 && p->after.y != 0 ) {
        idx = d_next_edge_coord.ptr(p->after.y)[p->after.x];
        if( idx != 0 ) {
            assert( idx >= 0 && idx < d_edgelist_2_sz );
            after = &d_edgelist_2[idx];
#ifndef NDEBUG
            inner_test_consistency( idx, after, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );

            if( p->after.x != after->coord.x || p->after.y != after->coord.y ) {
                printf("Intended coordinate is (%d,%d) at index %d, looked up coord is (%d,%d)\n",
                       p->after.x, p->after.y,
                       idx,
                       after->coord.x, after->coord.y );
            }
#endif // NDEBUG
        }
    }
    return after;
}

__device__
float inner_prod( const TriplePoint* l,
                  const TriplePoint* r,
                  cv::cuda::PtrStepSz16s d_dx,  // input
                  cv::cuda::PtrStepSz16s d_dy ) // input
{
    assert( l );
    assert( r );
    const int2& l_coord = l->coord;
    const int2& r_coord = r->coord;
    assert( l_coord.x >= 0 );
    assert( l_coord.x < d_dx.cols );
    assert( l_coord.y >= 0 );
    assert( l_coord.y < d_dx.rows );
    assert( r_coord.x >= 0 );
    assert( r_coord.x < d_dy.cols );
    assert( r_coord.y >= 0 );
    assert( r_coord.y < d_dy.rows );
    const float l_dx = d_dx.ptr(l_coord.y)[l_coord.x];
    const float l_dy = d_dy.ptr(l_coord.y)[l_coord.x];
    const float r_dx = d_dx.ptr(r_coord.y)[r_coord.x];
    const float r_dy = d_dy.ptr(r_coord.y)[r_coord.x];
    assert( l_dx != 0 || l_dy != 0 );
    assert( r_dx != 0 || r_dy != 0 );
    const float ret  = l_dx * r_dx + l_dy * r_dy;
    // assert( ret != 0 ); <- this can actually be 0 ?
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

/*
 * The voting procedure is modified from the original approach:
 * On entry:
 * - One kernel instance per TriplePoint. Each TriplePoint knows its own
 *   coordinates, as well as "before" and "after" coordinates from
 *   previous step
 * Step 1:
 * - Per each TriplePoint, find out whether it "chooses" a point
 *   This point is stored in its own "after" coordinate.
 * - If the point doesn't choose, set "after" to (0,0)
 * Step 2:
 * - All TriplePoints set "before" to (0,0)
 *   set both pointers to 0
 * Step 3:
 * - Sorting kernel?
 * - Create a new TriplePoint index list that contains only points
 *   with "after" that is not (0,0).
 * Step 4:
 * - New kernel, one kernel instance per TriplePoint (or non-null)
 *   TriplePoint
 * - For every point, find the TriplePoint indexed by "after" through
 *   the map of points.
 * - Use AtomicAdd to increase before.x of the "after" point by 1
 * - Use AtomicExch to chain self into "after" points' list of voters
 * Step 5:
 * - Sorting kernel
 * - Create a new TriplePoint index list that is sorted by before.x
 */

/* Brief: Voting procedure. For every edge points, construct the 1st order approximation 
 * of the field line passing through it which consists in a polygonal line whose
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
__device__ int exit_point_1 = 0;
__device__ int exit_point_2 = 0;
__device__ int exit_point_3 = 0;
__device__ int exit_point_4 = 0;
__device__ int exit_point_5 = 0;
__device__ int exit_point_6 = 0;
__device__ int exit_point_7 = 0;
__device__ int exit_point_8 = 0;
__device__ int exit_point_9 = 0;
__device__ int exit_point_10 = 0;
__device__ int exit_point_11 = 0;

__device__
const TriplePoint* inner( TriplePoint*           d_edgelist_2,     // input
                          uint32_t               d_edgelist_2_sz,  // input
                          cv::cuda::PtrStepSz16s d_dx,  // input
                          cv::cuda::PtrStepSz16s d_dy,  // input
                          cv::cuda::PtrStepSz32s d_next_edge_coord, // input
                          // cv::cuda::PtrStepSz32s d_next_edge_after, // input
                          // cv::cuda::PtrStepSz32s d_next_edge_befor, // input
                          size_t                 numCrowns,
                          float                  ratioVoting )
{
    int offset = threadIdx.x + blockIdx.x * 32;
    if( offset >= d_edgelist_2_sz ) {
        atomicAdd( &exit_point_1, 1 );
        return 0;
    }
    if( offset == 0 ) {
        /* special case: offset 0 is intentionally empty */
        atomicAdd( &exit_point_1, 1 );
        return 0;
    }

    const TriplePoint* p = &d_edgelist_2[offset];

    if( p == 0 ) return 0;

#ifndef NDEBUG
    inner_test_consistency( offset, p, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );
#endif // NDEBUG

    float dist; // scalar to compute the distance ratio

    const TriplePoint* current = vote::find_befor( p, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );
    // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
    if( not current ) {
        atomicAdd( &exit_point_2, 1 );
        return 0;
    }

    // To save all sub-segments length
    int       vDistSize = 0;
#ifndef NDEBUG
    const int vDistMax  = numCrowns * 2 - 1;
#endif // NDEBUG
    float     vDist[MAX_CROWNS * 2 - 1];
    int flagDist = 1;

    // Length of the reconstructed field line approximation between the two
    // extremities.
    float totalDistance = 0.0;

    // compute difference in subsequent gradients orientation
    float cosDiffTheta = -vote::inner_prod( p, current, d_dx, d_dy );
    if( cosDiffTheta < 0.0 ) {
        atomicAdd( &exit_point_3, 1 );
        return 0;
    }

    float lastDist = vote::distance( p, current ); // hypotf is CUDA float intrinsic for sqrt(pow2+pow2)
    vDist[vDistSize++] = lastDist;
    assert( vDistSize <= vDistMax );
        
    // Add the sub-segment length to the total distance.
    totalDistance += lastDist;

    const TriplePoint* chosen = 0;

    // Iterate over all crowns
    for( int i=1; i < numCrowns; ++i ) {
        chosen = 0;

        // First in the gradient direction
        const TriplePoint* target = vote::find_after( current, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );
        // No edge point was found in that direction
        if( not target ) {
            atomicAdd( &exit_point_4, 1 );
            return 0;
        }

        // Check the difference of two consecutive angles
        cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
        if( cosDiffTheta < 0.0 ) {
            atomicAdd( &exit_point_5, 1 );
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
            atomicAdd( &exit_point_6, 1 );
            return 0;
        }

        // lastDist = dist;
        current = target;
        // Second in the opposite gradient direction
        // target = vote::find_befor( current, d_next_edge_befor, d_edgelist_2 );
        target = vote::find_befor( current, d_next_edge_coord, d_edgelist_2, d_edgelist_2_sz );
        if( not target ) {
            atomicAdd( &exit_point_7, 1 );
            return 0;
        }

        cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
        if( cosDiffTheta < 0.0 ) {
            atomicAdd( &exit_point_8, 1 );
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
            atomicAdd( &exit_point_9, 1 );
            return 0;
        }

        // lastDist = dist;
        current = target;
        chosen = current;

        if( !current ) {
            atomicAdd( &exit_point_10, 1 );
            return 0;
        }
    }
    atomicAdd( &exit_point_11, 1 );
    return chosen;
}

} // namespace vote

__device__ int count_choices = 0;

__global__
void vote_kernel( TriplePoint*           d_edgelist_2,     // input
                  uint32_t               d_edgelist_2_sz,  // input
                  cv::cuda::PtrStepSz16s d_dx,  // input
                  cv::cuda::PtrStepSz16s d_dy,  // input
                  cv::cuda::PtrStepSz32s d_next_edge_coord, // input
                  // cv::cuda::PtrStepSz32s d_next_edge_after, // input
                  // cv::cuda::PtrStepSz32s d_next_edge_befor, // input
                  size_t                 numCrowns,
                  float                  ratioVoting )
{
    const TriplePoint* chosen = vote::inner( d_edgelist_2,     // input
                                             d_edgelist_2_sz,  // input
                                             d_dx,  // input
                                             d_dy,  // input
                                             d_next_edge_coord, // input
                                             // d_next_edge_after, // input
                                             // d_next_edge_befor, // input
                                             numCrowns,
                                             ratioVoting );
    if( chosen )
        atomicAdd( &count_choices, 1 );
}

__global__
void init_choices( )
{
    count_choices = 0;
    vote::exit_point_1 = 0;
    vote::exit_point_2 = 0;
    vote::exit_point_3 = 0;
    vote::exit_point_4 = 0;
    vote::exit_point_5 = 0;
    vote::exit_point_6 = 0;
    vote::exit_point_7 = 0;
    vote::exit_point_8 = 0;
    vote::exit_point_9 = 0;
    vote::exit_point_10 = 0;
    vote::exit_point_11 = 0;
}

__global__
void print_choices( )
{
    printf("The number of points chosen is %d\n", count_choices );
    printf("    Exit point  1: %d exits\n", vote::exit_point_1 );
    printf("    Exit point  2: %d exits\n", vote::exit_point_2 );
    printf("    Exit point  3: %d exits\n", vote::exit_point_3 );
    printf("    Exit point  4: %d exits\n", vote::exit_point_4 );
    printf("    Exit point  5: %d exits\n", vote::exit_point_5 );
    printf("    Exit point  6: %d exits\n", vote::exit_point_6 );
    printf("    Exit point  7: %d exits\n", vote::exit_point_7 );
    printf("    Exit point  8: %d exits\n", vote::exit_point_8 );
    printf("    Exit point  9: %d exits\n", vote::exit_point_9 );
    printf("    Exit point 10: %d exits\n", vote::exit_point_10 );
    printf("    Exit point 11: %d exits\n", vote::exit_point_11 );
    printf("    Sum exits: %d\n", vote::exit_point_2 + vote::exit_point_3 + vote::exit_point_4 + vote::exit_point_5 + vote::exit_point_6 + vote::exit_point_7 + vote::exit_point_8 + vote::exit_point_9 + vote::exit_point_10 + vote::exit_point_11 );
}

__host__
void Frame::applyVote( const cctag::Parameters& params )
{
    cout << "Enter " << __FUNCTION__ << endl;

    if( params._numCrowns > MAX_CROWNS ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter crowns is " << MAX_CROWNS << ", parameter file wants " << params._numCrowns << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

    dim3 block;
    dim3 grid;
    block.x = 32;
    block.y = 1;
    block.z = 1;
    grid.x  = _h_edgelist_2_sz / 32 + ( _h_edgelist_2_sz % 32 != 0 ? 1 : 0 );
    grid.y  = 1;
    grid.z  = 1;

    cout << "We have to check " << _h_edgelist_2_sz << " edges" << endl;

    init_choices<<<1,1,0,_stream>>>( );

    cout << "vote_kernel <<< (" << grid.x << "," << grid.y << "," << grid.z << "),(" << block.x << "," << block.y << "," << block.z << ") >>>" << endl;

    vote_kernel
        <<<grid,block,0,_stream>>>
        ( _d_edgelist_2,
          _h_edgelist_2_sz,
          _d_dx,
          _d_dy,
          _d_next_edge_coord,
          params._numCrowns,
          params._ratioVoting );
    POP_CHK_CALL_IFSYNC;

    print_choices<<<1,1,0,_stream>>>( );

    cout << "Leave " << __FUNCTION__ << endl;
}

#if 0
    // THIS MUST BE A SEPARATE STEP !

    bool is_a_seed = false;
    // Check if winner was found
    if (chosen) {
        // Associate winner with its voter (add the current point)
        winners[chosen].push_back(&p);

        // update flow length average scale factor
        chosen->_flowLength = (chosen->_flowLength * (winners[chosen].size() - 1) + totalDistance) / winners[chosen].size();

        // If chosen has a number of votes greater than one of
        // the edge points, then update max.
        if (winners[chosen].size() >= params._minVotesToSelectCandidate) {
            if (chosen->_isMax == -1) {
                seeds.push_back(chosen);
                is_a_seed = true;
            }
            chosen->_isMax = winners[chosen].size();
        }
    }
    return is_a_seed;
#endif

} // namespace popart

