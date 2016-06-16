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
#include "onoff.h"

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

/* functions cannot be __global__ __device__ */
__device__
inline void initChainedEdgeCoords_2( FrameMetaPtr& meta )
{
    /* Note: the initial _voters.dev.size is set to 1 because it is used
     * as an index for writing points into an array. Starting the counter
     * at 1 allows to distinguish unchained points (0) from chained
     * points non-0.
     */
    meta.list_size_voters() = 1;
}

__global__
void initChainedEdgeCoords( FrameMetaPtr meta )
{
    initChainedEdgeCoords_2( meta );
}

__device__
bool gradient_descent_inner( const CudaEdgePoint&         point,
                             int                          direction,
                             int2&                        out_edge_info,
                             const cv::cuda::PtrStepSzb   edge_image )
{
    // const int offset = blockIdx.x * 32 + threadIdx.x;
    // int direction    = threadIdx.y == 0 ? -1 : 1;
    // if( offset >= d_edgepoints.Size() ) return false;
    // const int idx = d_edgepoints.ptr[offset].x;
    // const int idy = d_edgepoints.ptr[offset].y;
#if 0
    /* This was necessary to allow the "after" threads (threadIdx.y==1)
     * to return sensible results even if "before" was 0.
     * Now useless, but kept just in case.  */
    out_edge_info.x = idx;
    out_edge_info.y = idy;
#endif
    const short2 id   = point._coord;
    const float2 grad = point._grad;

    assert( not outOfBounds( id.x, id.y, edge_image ) );
    if( outOfBounds( id.x, id.y, edge_image ) ) {
        return false; // should never happen
    }

    if( edge_image.ptr(id.y)[id.x] == 0 ) {
        assert( edge_image.ptr(id.y)[id.x] != 0 );
        return false; // should never happen
    }

    float  e     = 0.0f;
    float  dx    = direction * grad.x;
    float  dy    = direction * grad.y;

    assert( dx!=0 || dy!=0 );

    const float  adx   = d_abs( dx );
    const float  ady   = d_abs( dy );
    size_t n     = 0;
    int    stpX  = 0;
    int    stpY  = 0;
    int    x     = id.x;
    int    y     = id.y;
    
    if( ady > adx ) {
        updateXY(dy,dx,y,x,e,stpY,stpX);
    } else {
        updateXY(dx,dy,x,y,e,stpX,stpY);
    }
    n += 1;
    if ( dx*dx+dy*dy > tagParam.thrGradientMagInVote ) {
        const float dxRef = dx;
        const float dyRef = dy;
        const float dx2   = grad.x; // d_dx.ptr(id.y)[id.x];
        const float dy2   = grad.y; // d_dy.ptr(id.y)[id.x];
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
        out_edge_info = make_int2( x, y );
        assert( id.x != x || id.y != y );
        return true;
    }
    
    while( n <= tagParam.distSearch ) {
        if( ady > adx ) {
            updateXY(dy,dx,y,x,e,stpY,stpX);
        } else {
            updateXY(dx,dy,x,y,e,stpX,stpY);
        }
        n += 1;

        if( outOfBounds( x, y, edge_image ) ) return false;

        ret = edge_image.ptr(y)[x];
        if( ret ) {
            out_edge_info = make_int2( x, y );
            assert( id.x != x || id.y != y );
            return true;
        }

        if( ady > adx ) {
            if( outOfBounds( x, y - stpY, edge_image ) ) return false;

            ret = edge_image.ptr(y-stpY)[x];
            if( ret ) {
                out_edge_info = make_int2( x, y-stpY );
                assert( id.x != x || id.y != y-stpY );
                return true;
            }
        } else {
            if( outOfBounds( x - stpX, y, edge_image ) ) return false;

            ret = edge_image.ptr(y)[x-stpX];
            if( ret ) {
                out_edge_info = make_int2( x-stpX, y );
                assert( id.x != x-stpX || id.y != y );
                return true;
            }
        }
    }
    return false;
}

__global__
void gradient_descent( FrameMetaPtr                     meta,
                       const DevEdgeList<CudaEdgePoint> d_edgepoints, // input-output
                       const cv::cuda::PtrStepSzb       edge_image,
                       cv::cuda::PtrStepSz32s           d_edgepoint_map, // input
                       DevEdgeList<int>                 voters ) // output
{
    assert( blockDim.x * gridDim.x < meta.list_size_edgepoints() + 32 );
    assert( meta.list_size_voters() <= 2*meta.list_size_edgepoints() );

    int2   out_edge_info;
    bool   keep = false;
    // before -1  if threadIdx.y == 0
    // after   1  if threadIdx.y == 1

    const int  offset    = blockIdx.x * 32 + threadIdx.x;
    const bool in_bounds = ( offset < meta.list_size_edgepoints() );
    // cannot return if not in_bounds; __syncthreads would deadlock

    CudaEdgePoint& point = d_edgepoints.ptr[offset];

    if( in_bounds ) {
        const int direction = threadIdx.y == 0 ? -1 : 1;

        keep = gradient_descent_inner( point,
                                       direction,
                                       out_edge_info,
                                       edge_image );
    }

    __syncthreads();

    assert( not keep || not outOfBounds( out_edge_info.z, out_edge_info.w, edge_image ) );

    __shared__ int merge_directions[2][32];
    int edgepoint_index = d_edgepoint_map.ptr(out_edge_info.y)[out_edge_info.x];
    merge_directions[threadIdx.y][threadIdx.x] = keep ? edgepoint_index : -1;

    __syncthreads(); // be on the safe side: __ballot syncs only one warp, we have 2

    /* The vote.cpp procedure computes points for before and after, and stores all
     * info in one point. In the voting procedure, after is never processed when
     * before is false.
     * Consequently, we ignore after completely when before is already false.
     * Lots of idling cores; but the _inner has a bad loop, and we may run into it,
     * which would be worse.
     */
    if( threadIdx.y == 1 ) return;

    if( in_bounds ) {
        point._dev_befor = merge_directions[0][threadIdx.x];
        point._dev_after = merge_directions[1][threadIdx.x];
    }

    uint32_t mask = __ballot( keep );  // bitfield of warps with results

    // keep is false for all 32 threads
    if( mask == 0 ) return;

    uint32_t ct   = __popc( mask );    // horizontal reduce
    assert( ct <= 32 );

    int write_index = -1;
    if( threadIdx.x == 0 ) {
        // 0 gets warp's offset from global value and increases it
        // note that it is initialized with 1 to ensure that 0 represents a NULL pointer
        write_index = atomicAdd( &meta.list_size_voters(), (int)ct );

        if( meta.list_size_voters() > EDGE_POINT_MAX ) {
            printf( "max offset: (%d x %d)=%d\n"
                    "my  offset: (%d*32+%d)=%d\n"
                    "edges in:    %d\n"
                    "edges found: %d (total %d)\n",
                    gridDim.x, blockDim.x, blockDim.x * gridDim.x,
                    blockIdx.x, threadIdx.x, threadIdx.x + blockIdx.x*32,
                    meta.list_size_edgepoints(),
                    ct, meta.list_size_voters() );
            assert( meta.list_size_voters() <= 2*meta.list_size_edgepoints() );
        }
    }

    write_index = __shfl( write_index, 0 ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < EDGE_POINT_MAX ) {
        voters.ptr[write_index] = offset;
    }
}

#ifdef USE_SEPARABLE_COMPILATION_FOR_GRADDESC
__global__
void dp_call_01_gradient_descent(
    FrameMetaPtr                     meta,
    const DevEdgeList<CudaEdgePoint> d_edgepoints, // input-output
    const cv::cuda::PtrStepSzb       edge_image, // input
    cv::cuda::PtrStepSz32s           d_edgepoint_map, // input
    DevEdgeList<int>                 voters ) // output
{
    initChainedEdgeCoords_2( meta );

    /* No need to start more child kernels than the number of points found by
     * the Thinning stage.
     */
    int listsize = meta.list_size_edgepoints();

    /* The list of edge candidates is empty. Do nothing. */
    if( listsize == 0 ) return;

    dim3 block( 32, 2, 1 );
    dim3 grid( grid_divide( listsize, 32 ), 1, 1 );

    gradient_descent
        <<<grid,block>>>
        ( meta,
          d_edgepoints,         // input-output
          edge_image,
          d_edgepoint_map, // input 2D->edge index mapping
          voters ); // output
}
#endif // USE_SEPARABLE_COMPILATION_FOR_GRADDESC

} // namespace descent

#ifdef USE_SEPARABLE_COMPILATION_FOR_GRADDESC
__host__
bool Frame::applyDesc( )
{
    descent::dp_call_01_gradient_descent
        <<<1,1,0,_stream>>>
        ( _meta,                    // input-output
          _edgepoints.dev,      // input-output
          _d_edges,                 // input
          _d_edgepoint_map, // input 2D->edge index mapping
          _voters.dev );            // output
    POP_CHK_CALL_IFSYNC;
    return true;
}
#else // not USE_SEPARABLE_COMPILATION_FOR_GRADDESC
__host__
bool Frame::applyDesc( )
{
    descent::initChainedEdgeCoords
        <<<1,1,0,_stream>>>
        ( _meta );

    int listsize;

    // Note: right here, Dynamic Parallelism would avoid blocking.
    _meta.fromDevice( List_size_edgepoints, listsize, _stream );
    POP_CUDA_SYNC( _stream );

    if( listsize == 0 ) {
        cerr << "    I have not found any edges!" << endl;
        return false;
    }

    dim3 block( 32, 2, 1 );
    dim3 grid( grid_divide( listsize, 32 ), 1, 1 );

    descent::gradient_descent
        <<<grid,block,0,_stream>>>
        ( _meta,
          _edgepoints.dev,
          _d_edges,                 // input
          _d_edgepoint_map, // input 2D->edge index mapping
          _voters.dev );            // output - TriplePoints with before/after info
    POP_CHK_CALL_IFSYNC;

    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_GRADDESC


} // namespace popart

