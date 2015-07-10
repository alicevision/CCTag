#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <thrust/system/cuda/detail/cub/cub.cuh>
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
void inner_test_consistency( int                            p_idx,
                             const TriplePoint*             p,
                             cv::cuda::PtrStepSz32s         edgepoint_index_table,
                             const DevEdgeList<TriplePoint> chained_edgecoords )
{
    if( p == 0 ) {
        printf("Impossible bug, initialized from memory address\n");
        assert( 0 );
    }

    if( outOfBounds( p->coord, edgepoint_index_table ) ) {
        printf("Index (%d,%d) does not fit into coord lookup tables\n", p->coord.x, p->coord.y );
        assert( 0 );
    }

    int idx = edgepoint_index_table.ptr(p->coord.y)[p->coord.x];
    if( idx < 0 || idx >= chained_edgecoords.Size() ) {
        printf("Looked up index (coord) is out of bounds\n");
        assert( 0 );
    }

    if( idx != p_idx ) {
        printf("Looked up index %d is not identical to input index %d\n", idx, p_idx);
        assert( 0 );
    }

    if( outOfBounds( p->befor, edgepoint_index_table ) ) {
        printf("Before coordinations (%d,%d) do not fit into lookup tables\n", p->befor.x, p->befor.y );
        assert( 0 );
    }

    if( outOfBounds( p->after, edgepoint_index_table ) ) {
        printf("After coordinations (%d,%d) do not fit into lookup tables\n", p->after.x, p->after.y );
        assert( 0 );
    }
}
#endif // NDEBUG

__device__
inline
TriplePoint* find_befor( const TriplePoint*       p,
                         cv::cuda::PtrStepSz32s   edgepoint_index_table,
                         DevEdgeList<TriplePoint> chained_edgecoords )
{
    if( p->befor.x != 0 || p->befor.y != 0 ) {
        int idx = edgepoint_index_table.ptr(p->befor.y)[p->befor.x];
        if( idx != 0 ) {
            assert( idx >= 0 && idx < chained_edgecoords.Size() );
            TriplePoint* before = &chained_edgecoords.ptr[idx];
#ifndef NDEBUG
            inner_test_consistency( idx, before, edgepoint_index_table, chained_edgecoords );

            if( p->befor.x != before->coord.x || p->befor.y != before->coord.y ) {
                printf("Intended coordinate is (%d,%d) at index %d, looked up coord is (%d,%d)\n",
                       p->befor.x, p->befor.y,
                       idx,
                       before->coord.x, before->coord.y );
            }
#endif // NDEBUG
            return before;
        }
    }
    return 0;
}


__device__
inline
TriplePoint* find_after( const TriplePoint*             p,
                               cv::cuda::PtrStepSz32s   edgepoint_index_table,
                               DevEdgeList<TriplePoint> chained_edgecoords )
{
    TriplePoint* after = 0;
    int          idx    = 0;
    if( p->after.x != 0 && p->after.y != 0 ) {
        idx = edgepoint_index_table.ptr(p->after.y)[p->after.x];
        if( idx != 0 ) {
            assert( idx >= 0 && idx < chained_edgecoords.Size() );
            after = &chained_edgecoords.ptr[idx];
#ifndef NDEBUG
            inner_test_consistency( idx, after, edgepoint_index_table, chained_edgecoords );

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

__device__
void updateXY(const float & dx, const float & dy, int & x, int & y,  float & e, int & stpX, int & stpY)
{
    float d = dy / dx;
    float a = d_abs( d );
    // stpX = ( dx < 0 ) ? -1 : ( dx == 0 ) ? 0 : 1;
    // stpY = ( dy < 0 ) ? -1 : ( dy == 0 ) ? 0 : 1;
    // stpX = ( dx < 0 ) ? -1 : 1;
    // stpY = ( dy < 0 ) ? -1 : 1;
    stpX = d_sign( dx );
    stpY = d_sign( dy );
    e   += a;
    x   += stpX;
    if( e >= 0.5 ) {
        y += stpY;
        e -= 1.0f;
    }
}

__device__
bool gradient_descent_inner( int4&                  out_edge_info,
                             DevEdgeList<int2>      all_edgecoords,
                             cv::cuda::PtrStepSzb   edge_image,
                             uint32_t               nmax,
                             cv::cuda::PtrStepSz16s d_dx,
                             cv::cuda::PtrStepSz16s d_dy,
                             int32_t                thrGradient )
{
    const int offset = blockIdx.x * 32 + threadIdx.x;
    int direction    = threadIdx.y == 0 ? -1 : 1;

    if( offset >= all_edgecoords.Size() ) return false;

    const int idx = all_edgecoords.ptr[offset].x;
    const int idy = all_edgecoords.ptr[offset].y;

    if( outOfBounds( idx, idy, edge_image ) ) return false; // should never happen

    if( edge_image.ptr(idy)[idx] == 0 ) return false; // should never happen

    float  e     = 0.0f;
    float  dx    = direction * d_dx.ptr(idy)[idx];
    float  dy    = direction * d_dy.ptr(idy)[idx];

    assert( dx!=0 || dy!=0 );

    const float  adx   = d_abs( dx );
    const float  ady   = d_abs( dy );
    size_t n     = 0;
    int    stpX  = 0;
    int    stpY  = 0;
    int    x     = idx;
    int    y     = idy;
    
    if( ady > adx ) {
        updateXY(dy,dx,y,x,e,stpY,stpX);
    } else {
        updateXY(dx,dy,x,y,e,stpX,stpY);
    }
    n += 1;
    if ( dx*dx+dy*dy > thrGradient ) {
        const float dxRef = dx;
        const float dyRef = dy;
        const float dx2 = d_dx.ptr(idy)[idx];
        const float dy2 = d_dy.ptr(idy)[idx];
        const float compdir = dx2*dxRef+dy2*dyRef;
        // dir = ( compdir < 0 ) ? -1 : 1;
        direction = d_sign( compdir );
        dx = direction * dx2;
        dy = direction * dy2;
    }
    if( ady > adx ) {
        updateXY(dy,dx,y,x,e,stpY,stpX);
    } else {
        updateXY(dx,dy,x,y,e,stpX,stpY);
    }
    n += 1;

    if( outOfBounds( x, y, edge_image ) ) return false;

    uint8_t ret = edge_image.ptr(y)[x];
    if( ret ) {
        out_edge_info = make_int4( idx, idy, x, y );
        assert( idx != x || idy != y );
        return true;
    }
    
    while( n <= nmax ) {
        if( ady > adx ) {
            updateXY(dy,dx,y,x,e,stpY,stpX);
        } else {
            updateXY(dx,dy,x,y,e,stpX,stpY);
        }
        n += 1;

        if( outOfBounds( x, y, edge_image ) ) return false;

        ret = edge_image.ptr(y)[x];
        if( ret ) {
            out_edge_info = make_int4( idx, idy, x, y );
            assert( idx != x || idy != y );
            return true;
        }

        if( ady > adx ) {
            if( outOfBounds( x, y - stpY, edge_image ) ) return false;

            ret = edge_image.ptr(y-stpY)[x];
            if( ret ) {
                out_edge_info = make_int4( idx, idy, x, y-stpY );
                assert( idx != x || idy != y-stpY );
                return true;
            }
        } else {
            if( outOfBounds( x - stpX, y, edge_image ) ) return false;

            ret = edge_image.ptr(y)[x-stpX];
            if( ret ) {
                out_edge_info = make_int4( idx, idy, x-stpX, y );
                assert( idx != x-stpX || idy != y );
                return true;
            }
        }
    }
    return false;
}

__global__
void gradient_descent( DevEdgeList<int2>        all_edgecoords,
                       DevEdgeList<TriplePoint> chained_edgecoords,
                       cv::cuda::PtrStepSz32s   edgepoint_index_table,
                       uint32_t                 max_num_edges,
                       cv::cuda::PtrStepSzb     edge_image,
                       uint32_t                 nmax,
                       cv::cuda::PtrStepSz16s   d_dx,
                       cv::cuda::PtrStepSz16s   d_dy,
                       int32_t                  thrGradient )
{
    assert( blockDim.x * gridDim.x < all_edgecoords.Size() + 32 );
    assert( chained_edgecoords.Size() <= 2*all_edgecoords.Size() );

    int4 out_edge_info;
    bool keep;
    // before -1  if threadIdx.y == 0
    // after   1  if threadIdx.y == 1

    keep = gradient_descent_inner( out_edge_info,
                                   all_edgecoords,
                                   edge_image,
                                   nmax,
                                   d_dx,
                                   d_dy,
                                   thrGradient );

    __syncthreads();
    __shared__ int2 merge_directions[2][32];
    merge_directions[threadIdx.y][threadIdx.x].x = keep ? out_edge_info.z : 0;
    merge_directions[threadIdx.y][threadIdx.x].y = keep ? out_edge_info.w : 0;

    /* The vote.cpp procedure computes points for before and after, and stores all
     * info in one point. In the voting procedure, after is never processed when
     * before is false.
     * Consequently, we ignore after completely when before is already false.
     * Lots of idling cores; but the _inner has a bad loop, and we may run into it,
     * which would be worse.
     */
    if( threadIdx.y == 1 ) return;

    __syncthreads(); // be on the safe side: __ballot syncs only one warp, we have 2

    TriplePoint out_edge;
    out_edge.coord.x = keep ? out_edge_info.x : 0;
    out_edge.coord.y = keep ? out_edge_info.y : 0;
    out_edge.befor.x = keep ? merge_directions[0][threadIdx.x].x : 0;
    out_edge.befor.y = keep ? merge_directions[0][threadIdx.x].y : 0;
    out_edge.after.x = keep ? merge_directions[1][threadIdx.x].x : 0;
    out_edge.after.y = keep ? merge_directions[1][threadIdx.x].y : 0;
    out_edge.my_vote            = 0;
    out_edge.chosen_flow_length = 0.0f;
    out_edge._winnerSize        = 0;
    out_edge._flowLength        = 0.0f;

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    if( mask == 0 ) return;

    uint32_t ct   = __popc( mask );    // horizontal reduce
    assert( ct <= 32 );

#if 0
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
#else
    uint32_t leader = 0;
#endif
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        // not that it is initialized with 1 to ensure that 0 represents a NULL pointer
        write_index = atomicAdd( chained_edgecoords.size, (int)ct );

        if( chained_edgecoords.Size() > 2*all_edgecoords.Size() ) {
            printf( "max offset: (%d x %d)=%d\n"
                    "my  offset: (%d*32+%d)=%d\n"
                    "edges in:    %d\n"
                    "edges found: %d (total %d)\n",
                    gridDim.x, blockDim.x, blockDim.x * gridDim.x,
                    blockIdx.x, threadIdx.x, threadIdx.x + blockIdx.x*32,
                    all_edgecoords.Size(),
                    ct, chained_edgecoords.Size() );
            assert( chained_edgecoords.Size() <= 2*all_edgecoords.Size() );
        }
    }
    // assert( *chained_edgecoord_list_sz >= 2*all_edgecoord_list_sz );

    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < max_num_edges ) {
        assert( out_edge.coord.x != out_edge.befor.x || out_edge.coord.y != out_edge.befor.y );
        assert( out_edge.coord.x != out_edge.after.x || out_edge.coord.y != out_edge.after.y );
        assert( out_edge.befor.x != out_edge.after.x || out_edge.befor.y != out_edge.after.y );

        /* At this point we know that we will keep the point.
         * Obviously, pointer chains in CUDA are tricky, but we can use index
         * chains based on the element's offset index in chained_edgecoord_list.
         * We use atomic exchange for the chaining operation.
         * Actually, for coord, we don't have to do it because there is a unique
         * mapping kernel instance to coord.
         * The after table _d_next_edge_after, on the hand, may form a true
         * chain.
         */
        edgepoint_index_table.ptr(out_edge.coord.y)[out_edge.coord.x] = write_index;

        // int* p_after = &d_next_edge_after.ptr(out_edge.after.y)[out_edge.after.x];
        // out_edge.next_after = atomicExch( p_after, write_index );

        // int* p_befor = &d_next_edge_befor.ptr(out_edge.befor.y)[out_edge.befor.x];
        // out_edge.next_befor = atomicExch( p_befor, write_index );

        chained_edgecoords.ptr[write_index] = out_edge;
    }
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
const TriplePoint* inner( DevEdgeList<TriplePoint> chained_edgecoords,     // input
                          cv::cuda::PtrStepSz16s   d_dx,  // input
                          cv::cuda::PtrStepSz16s   d_dy,  // input
                          cv::cuda::PtrStepSz32s   edgepoint_index_table, // input
                          size_t                   numCrowns,
                          float                    ratioVoting )
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
    inner_test_consistency( offset, p, edgepoint_index_table, chained_edgecoords );
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
    float     vDist[MAX_CROWNS * 2 - 1];
    int flagDist = 1;

    // Length of the reconstructed field line approximation between the two
    // extremities.
    float totalDistance = 0.0;

    // compute difference in subsequent gradients orientation
    float cosDiffTheta = -vote::inner_prod( p, current, d_dx, d_dy );
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

        // Check the difference of two consecutive angles
        cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
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

        cosDiffTheta = -vote::inner_prod( target, current, d_dx, d_dy );
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

#ifndef NDEBUG
__device__ int count_choices = 0;

__global__
void init_choices( )
{
    count_choices = 0;
}

__global__
void print_choices( )
{
    printf("    The number of points chosen is %d\n", count_choices );
}
#endif // NDEBUG

__global__
void construct_line( DevEdgeList<TriplePoint> chained_edgecoords, // input
                     DevEdgeList<int>         edge_indices,       //output
                     int                      edge_index_max,
                     cv::cuda::PtrStepSz16s   d_dx,             // input
                     cv::cuda::PtrStepSz16s   d_dy,             // input
                     cv::cuda::PtrStepSz32s   edgepoint_index_table, // input
                     size_t                   numCrowns,
                     float                    ratioVoting )
{
    const TriplePoint* chosen = vote::inner( chained_edgecoords,     // input
                                             d_dx,             // input
                                             d_dy,             // input
                                             edgepoint_index_table, // input
                                             numCrowns,
                                             ratioVoting );
    int idx = 0;
    uint32_t mask   = __ballot( chosen != 0 );
    uint32_t ct     = __popc( mask );
    if( ct == 0 ) return;

    uint32_t write_index;
    if( threadIdx.x == 0 ) {
        write_index = atomicAdd( edge_indices.size, (int)ct );
    }
    write_index = __shfl( write_index, 0 );
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    if( chosen ) {
#ifndef NDEBUG
        atomicAdd( &count_choices, 1 );
#endif

        if( edge_indices.Size() < edge_index_max ) {
            idx = edgepoint_index_table.ptr(chosen->coord.y)[chosen->coord.x];
            edge_indices.ptr[write_index] = idx;
        }
    }
}

} // namespace vote

/* For all chosen inner points, compute the average flow length and the
 * number of voters, and store in the TriplePoint structure of the chosen
 * inner point.
 *
 * chained_edgecoords is the list of all edges with their chaining info.
 * edge_indices is a list of indices into that list, containing the sorted,
 * unique indices of chosen inner points.
 */
__global__
void vote_eval_chosen( DevEdgeList<TriplePoint> chained_edgecoords, // input
                       DevEdgeList<int>         edge_indices        // input
                     )
{
    uint32_t offset = threadIdx.x + blockIdx.x * 32;
    if( offset >= edge_indices.Size() ) {
        return;
    }

    const int    chosen_edge_index = edge_indices.ptr[offset];
    TriplePoint* chosen_edge = &chained_edgecoords.ptr[chosen_edge_index];
    int          winner_size = 0;
    float        flow_length = 0.0f;

    /* This loop looks dangerous, but it is actually faster than
     * a manually partially unrolled loop.
     */
    const int voter_list_size = chained_edgecoords.Size();
    for( int i=0; i<voter_list_size; i++ )
    // for( int i=0; i<chained_edgecoords.Size(); i++ )
    {
        if( chained_edgecoords.ptr[i].my_vote == chosen_edge_index ) {
            winner_size += 1;
            flow_length += chained_edgecoords.ptr[i].chosen_flow_length;
        }
    }
    chosen_edge->_winnerSize = winner_size;
    chosen_edge->_flowLength = flow_length / winner_size;
}

#ifndef NDEBUG
__global__
void print_edgelist( int*     d_edgelist_3,
                     uint32_t d_edgelist_3_sz )
{
    for( int i=0; i<d_edgelist_3_sz; i++ ) {
        printf( "  i=%d e[i]=%d\n", i, d_edgelist_3[i] );
    }
}

__global__
void print_edgelist_3( DevEdgeList<TriplePoint> chained_edgecoords,
                       int*                     d_edgelist_3,
                       uint32_t                 d_edgelist_3_sz )
{
    for( int i=0; i<d_edgelist_3_sz; i++ ) {
        int inner_point_index = d_edgelist_3[i];
        TriplePoint* chosen = &chained_edgecoords.ptr[inner_point_index];
        printf( "  i=%d e[i]=%d (%d,%d) has %d voters, %f avg len\n",
                i,
                inner_point_index,
                chosen->coord.x,
                chosen->coord.y,
                chosen->_winnerSize,
                chosen->_flowLength );
        for( int j=0; j<chained_edgecoords.Size(); j++ ) {
            const TriplePoint& voter = chained_edgecoords.ptr[j];
            if( voter.my_vote == inner_point_index ) {
                printf("    voter for %d: (%d,%d)\n", inner_point_index, voter.coord.x, voter.coord.y );
            }
        }
    }
}
#endif // NDEBUG

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

    CUB_RUNTIME_FUNCTION
    __host__ __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_array.ptr[a]._winnerSize >= _compare);
    }
};

__host__
bool Voting::gradientDescent( const cctag::Parameters&     params,
                              const cv::cuda::PtrStepSzb   edge_image,
                              const cv::cuda::PtrStepSz16s d_dx,
                              const cv::cuda::PtrStepSz16s d_dy,
                              cudaStream_t                 stream )
{
    cout << "  Enter " << __FUNCTION__ << endl;

    int listsize;

    // Note: right here, Dynamic Parallelism would avoid blocking.
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &listsize, _all_edgecoords.dev.size, sizeof(int), stream );
    POP_CUDA_SYNC( stream );

    if( listsize == 0 ) {
        cerr << "    I have not found any edges!" << endl;
        cerr << "  Leave " << __FUNCTION__ << endl;
        return false;
    }

    const uint32_t nmax          = params._distSearch;
    const int32_t  threshold     = params._thrGradientMagInVote;
    dim3           block;
    dim3           grid;
    block.x = 32;
    block.y = 2;
    block.z = 1;
    grid.x  = listsize / 32 + ( listsize % 32 != 0 ? 1 : 0 );
    grid.y  = 1;
    grid.z  = 1;

    /* Note: the initial _chained_edgecoords.dev.size is set to 1 because it is used
     * as an index for writing points into a array. Starting the counter
     * at 1 allows to distinguish unchained points (0) from chained
     * points non-0.
     */
    POP_CUDA_SETX_ASYNC( _chained_edgecoords.dev.size, (int)1, stream );

#ifndef NDEBUG
    cout << "    calling gradient descent with " << listsize << " edge points" << endl;
    cout << "    max num edges is " << params._maxEdges << endl;

    cout << "    grid (" << grid.x << "," << grid.y << "," << grid.z << ")"
         << " block (" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif // NDEBUG

    vote::gradient_descent
        <<<grid,block,0,stream>>>
        ( _all_edgecoords.dev,
          _chained_edgecoords.dev,
          _d_edgepoint_index_table,
          params._maxEdges,
          edge_image, nmax, d_dx, d_dy, threshold );
    POP_CHK_CALL_IFSYNC;

    cout << "  Leave " << __FUNCTION__ << endl;
    return true;
}

__host__
bool Voting::constructLine( const cctag::Parameters&     params,
                            const cv::cuda::PtrStepSz16s d_dx,
                            const cv::cuda::PtrStepSz16s d_dy,
                            cudaStream_t                 stream )
{
    cout << "  Enter " << __FUNCTION__ << endl;

    // Note: right here, Dynamic Parallelism would avoid blocking.
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_chained_edgecoords.host.size,
                                   _chained_edgecoords.dev.size,
                                   sizeof(int), stream );
    POP_CUDA_SYNC( stream );

    int listsize = _chained_edgecoords.host.size;

    cout << "    after gradient descent, edge counter is " << listsize << endl;

    if( listsize == 0 ) {
        cout << "  Leave " << __FUNCTION__ << endl;
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

    POP_CUDA_SET0_ASYNC( _edge_indices.dev.size, stream );

#ifndef NDEBUG
    vote::init_choices<<<1,1,0,stream>>>( );
#endif // NDEBUG

    vote::construct_line
        <<<grid,block,0,stream>>>
        ( _chained_edgecoords.dev,
          _edge_indices.dev,
          params._maxEdges,
          d_dx,
          d_dy,
          _d_edgepoint_index_table,
          params._numCrowns,
          params._ratioVoting );
    POP_CHK_CALL_IFSYNC;

#ifndef NDEBUG
    vote::print_choices<<<1,1,0,stream>>>( );
    POP_CHK_CALL_IFSYNC;
#endif // NDEBUG

    cout << "  Leave " << __FUNCTION__ << endl;
    return true;
}

__host__
void Frame::applyVote( const cctag::Parameters& params )
{
    cout << "Enter " << __FUNCTION__ << endl;

    if( params._numCrowns > MAX_CROWNS ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter crowns is " << MAX_CROWNS
             << ", parameter file wants " << params._numCrowns << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

    bool success;
    
    success = _vote.gradientDescent( params,
                                     _d_edges,
                                     _d_dx,
                                     _d_dy,
                                     _stream );

    if( not success ) {
        cout << "Leave " << __FUNCTION__ << endl;
        return;
    }

    success = _vote.constructLine( params,
                                   _d_dx,
                                   _d_dy,
                                   _stream );

    if( not success ) {
        cout << "Leave " << __FUNCTION__ << endl;
        return;
    }

    /* For every chosen, compute the average flow size from all
     * of its voters, and count the number of its voters.
     */
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._edge_indices.host.size, _vote._edge_indices.dev.size, sizeof(int), _stream );
    POP_CUDA_SYNC( _stream );

    if( _vote._edge_indices.host.size > 0 ) {
        /* Note: we use the intermediate picture plane, _d_intermediate, as assist
         *       buffer for CUB algorithms. It is extremely likely that this plane
         *       is large enough in all cases. If there are any problems, call
         *       the function with assist_buffer=0, and the function will return
         *       the required size in assist_buffer_sz (call by reference).
         */
        void*  assist_buffer = (void*)_d_intermediate.data;
        size_t assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        cub::DoubleBuffer<int> d_keys( _vote._edge_indices.dev.ptr,
                                       _vote._edge_indices_2.dev.ptr );

        /* After SortKeys, both buffers in d_keys have been altered.
         * The final result is stored in d_keys.d_buffers[d_keys.selector].
         * The other buffer is invalid.
         */
        cub::DeviceRadixSort::SortKeys( assist_buffer,
                                        assist_buffer_sz,
                                        d_keys,
                                        _vote._edge_indices.host.size,
                                        0,             // begin_bit
                                        sizeof(int)*8, // end_bit
                                        _stream );
        POP_CHK_CALL_IFSYNC;

        if( d_keys.d_buffers[d_keys.selector] == _vote._edge_indices_2.dev.ptr ) {
            std::swap( _vote._edge_indices.dev.ptr,   _vote._edge_indices_2.dev.ptr );
            std::swap( _vote._edge_indices.dev.size,  _vote._edge_indices_2.dev.size );
            std::swap( _vote._edge_indices.host.size, _vote._edge_indices_2.host.size );
        }

        // safety: SortKeys is allowed to alter assist_buffer_sz
        assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        /* Unique ensure that we check every "chosen" point only once.
         * Output is in _vote._edge_indices_2.dev
         */
        cub::DeviceSelect::Unique( assist_buffer,
                                   assist_buffer_sz,
                                   _vote._edge_indices.dev.ptr,     // input
                                   _vote._edge_indices_2.dev.ptr,   // output
                                   _vote._edge_indices_2.dev.size,  // output
                                   _vote._edge_indices_2.host.size, // input (unchanged in sort)
                                   _stream );
        POP_CHK_CALL_IFSYNC;

        /* Without Dynamic Parallelism, we must block here to retrieve the
         * value d_num_selected_out from the device before the voting
         * step.
         */
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._edge_indices_2.host.size,
                                       _vote._edge_indices_2.dev.size,
                                       sizeof(int), _stream );
        POP_CUDA_SYNC( _stream );

        // int num_selected_out = _vote._edge_indices_2.host.size;

        dim3 block;
        dim3 grid;

        block.x = 32;
        block.y = 1;
        block.z = 1;
        grid.x  = _vote._edge_indices_2.host.size / 32 + ( _vote._edge_indices_2.host.size % 32 != 0 ? 1 : 0 );
        grid.y  = 1;
        grid.z  = 1;

        /* Add number of voters to chosen inner points, and
         * add average flow length to chosen inner points.
         */
        vote_eval_chosen
            <<<grid,block,0,_stream>>>
            ( _vote._chained_edgecoords.dev,
              _vote._edge_indices_2.dev );
        POP_CHK_CALL_IFSYNC;

        // safety: SortKeys is allowed to alter assist_buffer_sz
        assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;

        /* Filter all chosen inner points that have fewer
         * voters than required by Parameters.
         */
        NumVotersIsGreaterEqual select_op( params._minVotesToSelectCandidate,
                                           _vote._chained_edgecoords.dev );
        cub::DeviceSelect::If( assist_buffer,
                               assist_buffer_sz,
                               _vote._edge_indices_2.dev.ptr,
                               _vote._edge_indices.dev.ptr,
                               _vote._edge_indices.dev.size,
                               _vote._edge_indices_2.host.size,
                               select_op,
                               _stream );
        POP_CHK_CALL_IFSYNC;

        /* Without Dynamic Parallelism, we must block here to retrieve the
         * value d_num_selected_out from the device before the voting
         * step.
         */
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._edge_indices.host.size, _vote._edge_indices.dev.size, sizeof(int), _stream );
        POP_CUDA_SYNC( _stream );

        cout << "  Number of viable inner points: " << _vote._edge_indices.host.size << endl;

#ifndef NDEBUG
#if 0
        print_edgelist_3<<<1,1,0,_stream>>>( _vote._chained_edgecoords.dev,
                                             _vote._edge_indices.dev.ptr,
                                             _vote._edge_indices.host.size );
        POP_CHK_CALL_IFSYNC;
#endif // 0
#endif // NDEBUG
    }
    cout << "Leave " << __FUNCTION__ << endl;
}

} // namespace popart

