#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <thrust/system/cuda/detail/cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

namespace descent
{

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
                             short2&                out_edge_d,
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
#if 0
    /* This was necessary to allow the "after" threads (threadIdx.y==1)
     * to return sensible results even if "before" was 0.
     * Now useless, but kept just in case.  */
    out_edge_info.x = idx;
    out_edge_info.y = idy;
#endif

    if( outOfBounds( idx, idy, edge_image ) ) return false; // should never happen

    if( edge_image.ptr(idy)[idx] == 0 ) {
        assert( edge_image.ptr(idy)[idx] != 0 );
        return false; // should never happen
    }

    float  e     = 0.0f;
    out_edge_d.x = d_dx.ptr(idy)[idx];
    out_edge_d.y = d_dy.ptr(idy)[idx];
    float  dx    = direction * out_edge_d.x;
    float  dy    = direction * out_edge_d.y;

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
        const float dx2 = out_edge_d.x; // d_dx.ptr(idy)[idx];
        const float dy2 = out_edge_d.y; // d_dy.ptr(idy)[idx];
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
                       DevEdgeList<TriplePoint> chained_edgecoords,    // output
                       cv::cuda::PtrStepSz32s   edgepoint_index_table, // output
                       uint32_t                 max_num_edges,
                       cv::cuda::PtrStepSzb     edge_image,
                       uint32_t                 nmax,
                       cv::cuda::PtrStepSz16s   d_dx,
                       cv::cuda::PtrStepSz16s   d_dy,
                       int32_t                  thrGradient )
{
    assert( blockDim.x * gridDim.x < all_edgecoords.Size() + 32 );
    assert( chained_edgecoords.Size() <= 2*all_edgecoords.Size() );

    int4   out_edge_info;
    short2 out_edge_d;
    bool   keep;
    // before -1  if threadIdx.y == 0
    // after   1  if threadIdx.y == 1

    keep = gradient_descent_inner( out_edge_info,
                                   out_edge_d,
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
    out_edge.d.x     = keep ? out_edge_d.x : 0;
    out_edge.d.y     = keep ? out_edge_d.y : 0;
    out_edge.descending.befor.x = keep ? merge_directions[0][threadIdx.x].x : 0;
    out_edge.descending.befor.y = keep ? merge_directions[0][threadIdx.x].y : 0;
    out_edge.descending.after.x = keep ? merge_directions[1][threadIdx.x].x : 0;
    out_edge.descending.after.y = keep ? merge_directions[1][threadIdx.x].y : 0;
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
        assert( out_edge.coord.x != out_edge.descending.befor.x ||
                out_edge.coord.y != out_edge.descending.befor.y );
        assert( out_edge.coord.x != out_edge.descending.after.x ||
                out_edge.coord.y != out_edge.descending.after.y );
        assert( out_edge.descending.befor.x != out_edge.descending.after.x ||
                out_edge.descending.befor.y != out_edge.descending.after.y );

        /* At this point we know that we will keep the point.
         * Obviously, pointer chains in CUDA are tricky, but we can use index
         * chains based on the element's offset index in chained_edgecoord_list.
         */
        edgepoint_index_table.ptr(out_edge.coord.y)[out_edge.coord.x] = write_index;

        chained_edgecoords.ptr[write_index] = out_edge;
    }
}

} // namespace descent

__host__
bool Voting::gradientDescent( const cctag::Parameters&     params,
                              const cv::cuda::PtrStepSzb   edge_image,
                              const cv::cuda::PtrStepSz16s d_dx,
                              const cv::cuda::PtrStepSz16s d_dy,
                              cudaStream_t                 stream )
{
    // cout << "  Enter " << __FUNCTION__ << endl;

    int listsize;

    // Note: right here, Dynamic Parallelism would avoid blocking.
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &listsize, _all_edgecoords.dev.size, sizeof(int), stream );
    POP_CUDA_SYNC( stream );

    if( listsize == 0 ) {
        cerr << "    I have not found any edges!" << endl;
        // cerr << "  Leave " << __FUNCTION__ << endl;
        return false;
    }

    const uint32_t nmax          = params._distSearch;
    const int32_t  threshold     = params._thrGradientMagInVote;
    dim3           block;
    dim3           grid;
    block.x = 32;
    block.y = 2;
    block.z = 1;
    grid.x  = grid_divide( listsize, 32 );
    grid.y  = 1;
    grid.z  = 1;

    /* Note: the initial _chained_edgecoords.dev.size is set to 1 because it is used
     * as an index for writing points into an array. Starting the counter
     * at 1 allows to distinguish unchained points (0) from chained
     * points non-0.
     */
    POP_CUDA_SETX_ASYNC( _chained_edgecoords.dev.size, (int)1, stream );

#ifndef NDEBUG
    debugPointIsOnEdge( edge_image, _all_edgecoords, stream );
#endif

#ifndef NDEBUG
#if 0
    cout << "    calling gradient descent with " << listsize << " edge points" << endl;
    cout << "    max num edges is " << params._maxEdges << endl;

    cout << "    Config: grid=" << grid << " block=" << block << endl;
#endif
#endif // NDEBUG

    descent::gradient_descent
        <<<grid,block,0,stream>>>
        ( _all_edgecoords.dev,
          _chained_edgecoords.dev,  // output - TriplePoints with before/after info
          _d_edgepoint_index_table, // output - table, map coord to TriplePoint index
          params._maxEdges,
          edge_image, nmax, d_dx, d_dy, threshold );
    POP_CHK_CALL_IFSYNC;

    // cout << "  Leave " << __FUNCTION__ << endl;
    return true;
}

__host__
bool Frame::applyDesc( const cctag::Parameters& params )
{
    // cout << "Enter " << __FUNCTION__ << endl;

    if( params._nCrowns > RESERVE_MEM_MAX_CROWNS ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter crowns is "
             << RESERVE_MEM_MAX_CROWNS
             << ", parameter file wants " << params._nCrowns << endl
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
        // cout << "Leave " << __FUNCTION__ << endl;
        return false;
    }
    // cout << "Leave " << __FUNCTION__ << endl;
#ifdef NDEBUG
    return true;
#else // NDEBUG
#ifdef DEBUG_RETURN_AFTER_GRADIENT_DESCENT
    return false;
#else  // DEBUG_RETURN_AFTER_GRADIENT_DESCENT
    return true;
#endif // DEBUG_RETURN_AFTER_GRADIENT_DESCENT
#endif // NDEBUG
}

} // namespace popart

