#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"

#define MAX_CROWNS  5

namespace popart
{

namespace vote
{

__device__
TriplePoint* find_befor( TriplePoint* p,
                         cv::cuda::PtrStepSz32s d_next_edge_befor,
                         TriplePoint*           d_edgelist_2 )
{
    const int2& coord = p->befor;
    if( coord.x == 0 && coord.y == 0 ) return 0;

    int          idx = d_next_edge_befor.ptr(coord.y)[coord.x];
    TriplePoint* b   = &d_edgelist_2[idx];
    return b;
}

__device__
TriplePoint* find_after( TriplePoint* p,
                         cv::cuda::PtrStepSz32s d_next_edge_after,
                         TriplePoint*           d_edgelist_2 )
{
    const int2& coord = p->after;
    if( coord.x == 0 && coord.y == 0 ) return 0;

    int          idx = d_next_edge_after.ptr(coord.y)[coord.x];
    TriplePoint* b   = &d_edgelist_2[idx];
    return b;
}

__device__
float inner_prod( TriplePoint* l,
                  TriplePoint* r,
                  cv::cuda::PtrStepSz16s d_dx,  // input
                  cv::cuda::PtrStepSz16s d_dy ) // input
{
    int2& l_coord = l->coord;
    int2& r_coord = r->coord;
    float l_dx = d_dx.ptr(l_coord.y)[l_coord.x];
    float l_dy = d_dy.ptr(l_coord.y)[l_coord.x];
    float r_dx = d_dx.ptr(r_coord.y)[r_coord.x];
    float r_dy = d_dy.ptr(r_coord.y)[r_coord.x];
    float ret  = l_dx * r_dx + l_dy * r_dy;
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

} // namespace vote

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
__device__
TriplePoint* vote_inner( TriplePoint* d_edgelist_2,     // input
                         uint32_t     d_edgelist_2_sz,  // input
                         cv::cuda::PtrStepSz16s d_dx,  // input
                         cv::cuda::PtrStepSz16s d_dy,  // input
                         cv::cuda::PtrStepSz32s d_next_edge_coord, // input
                         cv::cuda::PtrStepSz32s d_next_edge_after, // input
                         cv::cuda::PtrStepSz32s d_next_edge_befor, // input
                         size_t                 numCrowns,
                         float                  ratioVoting )
{
    int offset = threadIdx.x + blockIdx.x * 32;
    if( offset > d_edgelist_2_sz ) return 0;

    TriplePoint* p = &d_edgelist_2[offset];

    float dist; // scalar to compute the distance ratio

    // Alternate from the edge point found in the direction opposed to the gradient
    // direction.
    TriplePoint* current = vote::find_befor( p, d_next_edge_befor, d_edgelist_2 );
    // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
    TriplePoint* chosen = NULL;

    // To save all sub-segments length
    int       vDistSize = 0;
    const int vDistMax  = numCrowns * 2 - 1;
    float     vDist[MAX_CROWNS * 2 - 1];
    int flagDist = 1;

    // Length of the reconstructed field line approximation between the two
    // extremities.
    float totalDistance = 0.0;

    // compute difference in subsequent gradients orientation
    float cosDiffTheta = -vote::inner_prod( p, current, d_dx, d_dy );
    if( cosDiffTheta >= 0.0 ) {
        float lastDist = vote::distance( p, current ); // hypotf is CUDA float intrinsic for sqrt(pow2+pow2)
        vDist[vDistSize++] = lastDist;
        assert( vDistSize <= vDistMax );
        
        // Add the sub-segment length to the total distance.
        totalDistance += lastDist;

        // Iterate over all crowns
        for( int i=1; i < numCrowns; ++i ) {
            chosen = NULL;

            // First in the gradient direction
            TriplePoint* target = vote::find_after( current, d_next_edge_after, d_edgelist_2 );
            // No edge point was found in that direction
            if( not target ) {
                break;
            }
            
            // Check the difference of two consecutive angles
            cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
            if( cosDiffTheta < 0.0 ) {
                break;
            } else {
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
                    break;
                } else {
                    // lastDist = dist;
                    current = target;
                    // Second in the opposite gradient direction
                    target = vote::find_befor( current, d_next_edge_befor, d_edgelist_2 );
                    if( not target ) {
                        break;
                    }
                    cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
                    if( cosDiffTheta < 0.0 ) {
                        break;
                    } else {
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
                            break;
                        } else {
                            // lastDist = dist;
                            current = target;
                            chosen = current;
                            if (!current) {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return chosen;
}

#if 0
__host__
void Frame::vote( ...
                  const Parameters & params )
{
    if( params._numCrowns > MAX_CROWNS ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter crowns is " << MAX_CROWNS << ", parameter file wants " << params._numCrowns << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

}
#endif

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

