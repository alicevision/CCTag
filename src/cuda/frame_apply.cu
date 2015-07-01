#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

namespace popart
{

using namespace std;

static unsigned char h_thinning_lut[256] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
};

// Note that the transposed h_thinning_lut_t is not really necessary
// because flipping the 4 LSBs and 4 HSBs in the unsigned char that
// I use for lookup is really quick. Therefore: remove soon.
static unsigned char h_thinning_lut_t[256] = {
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
};

__device__ __constant__ unsigned char d_thinning_lut[256];

__device__ __constant__ unsigned char d_thinning_lut_t[256];

__global__
void compute_mag_l1( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * 32;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    dx = d_abs( dx );
    dy = d_abs( dy );
    dst.ptr(idy)[idx] = dx + dy;
}

__global__
void compute_mag_l2( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * 32;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    // --- hypot --
    dx *= dx;
    dy *= dy;
    dst.ptr(idy)[idx] = __fsqrt_rz( (float)( dx + dy ) );
}

__global__
void compute_map( const cv::cuda::PtrStepSz16s dx,
                  const cv::cuda::PtrStepSz16s dy,
                  const cv::cuda::PtrStepSz32u mag,
                  cv::cuda::PtrStepSzb         map,
                  const float                  low_thresh,
                  const float                  high_thresh )
{
    const int CANNY_SHIFT = 15;
    const int TG22 = (int32_t)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dx ) ) return;

    int32_t  dxVal  = dx.ptr(idy)[idx];
    int32_t  dyVal  = dy.ptr(idy)[idx];
    uint32_t magVal = mag.ptr(idy)[idx];

    // -1 if only is negative, 1 else
    // const int32_t signVal = (dxVal ^ dyVal) < 0 ? -1 : 1;
    const int32_t signVal = d_sign( dxVal ^ dyVal );

    dxVal = d_abs( dxVal );
    dyVal = d_abs( dyVal );

    // 0 - the pixel can not belong to an edge
    // 1 - the pixel might belong to an edge
    // 2 - the pixel does belong to an edge
    uint8_t edge_type = 0;

    if( magVal > low_thresh )
    {
        const int32_t tg22x = dxVal * TG22;
        const int32_t tg67x = tg22x + ((dxVal + dxVal) << CANNY_SHIFT);

        dyVal <<= CANNY_SHIFT;

        int2 x = (dyVal < tg22x) ? make_int2( idx - 1, idx + 1 )
                                 : (dyVal > tg67x ) ? make_int2( idx, idx )
                                                    : make_int2( idx - signVal, idx + signVal );
        int2 y = (dyVal < tg22x) ? make_int2( idy, idy )
                                 : make_int2( idy - 1, idy + 1 );

        x.x = clamp( x.x, dx.cols );
        x.y = clamp( x.y, dx.cols );
        y.x = clamp( y.x, dx.rows );
        y.y = clamp( y.y, dx.rows );

        if( magVal > mag.ptr(y.x)[x.x] && magVal >= mag.ptr(y.y)[x.y] ) {
            edge_type = 1 + (uint8_t)(magVal > high_thresh);
        }
    }
    __syncthreads();

    map.ptr(idy)[idx] = edge_type;
}

__device__
bool thinning_inner( const int idx, const int idy, cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, bool first_run )
{
    if( src.ptr(idy)[idx] == 0 ) {
        dst.ptr(idy)[idx] = 0;
        return false;
    }

    if( idx >= 1 && idy >=1 && idx <= src.cols-2 && idy <= src.rows-2 ) {
        uint8_t log = 0;

        log |= ( src.ptr(idy-1)[idx  ] != 0 ) ? 0x01 : 0;
        log |= ( src.ptr(idy-1)[idx+1] != 0 ) ? 0x02 : 0;
        log |= ( src.ptr(idy  )[idx+1] != 0 ) ? 0x04 : 0;
        log |= ( src.ptr(idy+1)[idx+1] != 0 ) ? 0x08 : 0;
        log |= ( src.ptr(idy+1)[idx  ] != 0 ) ? 0x10 : 0;
        log |= ( src.ptr(idy+1)[idx-1] != 0 ) ? 0x20 : 0;
        log |= ( src.ptr(idy  )[idx-1] != 0 ) ? 0x40 : 0;
        log |= ( src.ptr(idy-1)[idx-1] != 0 ) ? 0x80 : 0;

#if 1
        if( first_run )
            dst.ptr(idy)[idx] = d_thinning_lut[log];
        else
            dst.ptr(idy)[idx] = d_thinning_lut_t[log];
#else
        if( first_run == false ) {
            uint8_t b = log;
            b   = ( b   << 4 ) & 0xf0;
            log = ( ( log >> 4 ) & 0x0f ) | b;
        }

        dst.ptr(idy)[idx] = d_thinning_lut[log];
#endif
        return true;
    }
    return false;
}

__global__
void thinning( cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst )
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    thinning_inner( idx, idy, src, dst, true );
}

__global__
void thinning_and_store( cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, uint32_t* edgeCounter, uint32_t edgeMax, int2* edgeCoords )
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = thinning_inner( idx, idy, src, dst, false );

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( edgeCounter, ct );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < edgeMax ) {
        edgeCoords[write_index] = make_int2( idx, idy );
    }
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
                             int2*                  d_edgelist,
                             uint32_t               edgelist_sz,
                             cv::cuda::PtrStepSzb   edges,
                             // int                    direction,
                             uint32_t               nmax,
                             cv::cuda::PtrStepSz16s d_dx,
                             cv::cuda::PtrStepSz16s d_dy,
                             int32_t                thrGradient )
{
    const int offset = blockIdx.x * 32 + threadIdx.x;
    int direction    = threadIdx.y == 0 ? -1 : 1;

    if( offset >= edgelist_sz ) return false;

    const int idx = d_edgelist[offset].x;
    const int idy = d_edgelist[offset].y;
    // const int block_x = blockIdx.x * 32;
    // const int idx     = block_x + threadIdx.x;
    // const int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, edges ) ) return false; // should never happen

    if( edges.ptr(idy)[idx] == 0 ) return false; // should never happen

    float  e     = 0.0f;
    float  dx    = direction * d_dx.ptr(idy)[idx];
    float  dy    = direction * d_dy.ptr(idy)[idx];

#if 1
    assert( dx!=0 || dy!=0 );
#endif

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

    if( outOfBounds( x, y, edges ) ) return false;

    uint8_t ret = edges.ptr(y)[x];
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

        if( outOfBounds( x, y, edges ) ) return false;

        ret = edges.ptr(y)[x];
        if( ret ) {
            out_edge_info = make_int4( idx, idy, x, y );
            assert( idx != x || idy != y );
            return true;
        }

        if( ady > adx ) {
            if( outOfBounds( x, y - stpY, edges ) ) return false;

            ret = edges.ptr(y-stpY)[x];
            if( ret ) {
                out_edge_info = make_int4( idx, idy, x, y-stpY );
                assert( idx != x || idy != y-stpY );
                return true;
            }
        } else {
            if( outOfBounds( x - stpX, y, edges ) ) return false;

            ret = edges.ptr(y)[x-stpX];
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
void gradient_descent( int2*                  d_edgelist,
                       uint32_t               d_edgelist_sz,
                       TriplePoint*           d_new_edgelist,
                       uint32_t*              d_new_edgelist_sz,
                       cv::cuda::PtrStepSz32s d_next_edge_coord,
                       // cv::cuda::PtrStepSz32s d_next_edge_after,
                       // cv::cuda::PtrStepSz32s d_next_edge_befor,
                       uint32_t               max_num_edges,
                       cv::cuda::PtrStepSzb   edges,
                       uint32_t               nmax,
                       cv::cuda::PtrStepSz16s d_dx,
                       cv::cuda::PtrStepSz16s d_dy,
                       int32_t                thrGradient )
{
    assert( blockDim.x * gridDim.x < d_edgelist_sz + 32 );
    assert( *d_new_edgelist_sz <= 2*d_edgelist_sz );

    int4 out_edge_info;
    bool keep;
    // before -1  if threadIdx.y == 0
    // after   1  if threadIdx.y == 1

    keep = gradient_descent_inner( out_edge_info,
                                   d_edgelist,
                                   d_edgelist_sz,
                                   edges,
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

    /* This is an addition to the logic by griff, because I have trouble believing
     * that we have found a good gradient if searching in both directions leads
     * to identical coordinates.
     * @Lilian - please explain what happens here
     */
    // if( out_edge.befor.x == out_edge.after.x && out_edge.befor.y == out_edge.after.y ) keep = false;

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
        write_index = atomicAdd( d_new_edgelist_sz, ct );

        if( *d_new_edgelist_sz > 2*d_edgelist_sz ) {
            printf( "max offset: (%d x %d)=%d\n"
                    "my  offset: (%d*32+%d)=%d\n"
                    "edges in:    %d\n"
                    "edges found: %d (total %d)\n",
                    gridDim.x, blockDim.x, blockDim.x * gridDim.x,
                    blockIdx.x, threadIdx.x, threadIdx.x + blockIdx.x*32,
                    d_edgelist_sz,
                    ct, d_new_edgelist_sz );
            assert( *d_new_edgelist_sz <= 2*d_edgelist_sz );
        }
    }
    // assert( *d_new_edgelist_sz >= 2*d_edgelist_sz );

    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < max_num_edges ) {
        assert( out_edge.coord.x != out_edge.befor.x || out_edge.coord.y != out_edge.befor.y );
        assert( out_edge.coord.x != out_edge.after.x || out_edge.coord.y != out_edge.after.y );
        assert( out_edge.befor.x != out_edge.after.x || out_edge.befor.y != out_edge.after.y );

        /* At this point we know that we will keep the point.
         * Obviously, pointer chains in CUDA are tricky, but we can use index
         * chains based on the element's offset index in d_new_edgelist.
         * We use atomic exchange for the chaining operation.
         * Actually, for coord, we don't have to do it because there is a unique
         * mapping kernel instance to coord.
         * The after table _d_next_edge_after, on the hand, may form a true
         * chain.
         */
        d_next_edge_coord.ptr(out_edge.coord.y)[out_edge.coord.x] = write_index;

        // int* p_after = &d_next_edge_after.ptr(out_edge.after.y)[out_edge.after.x];
        // out_edge.next_after = atomicExch( p_after, write_index );

        // int* p_befor = &d_next_edge_befor.ptr(out_edge.befor.y)[out_edge.befor.x];
        // out_edge.next_befor = atomicExch( p_befor, write_index );

        d_new_edgelist[write_index] = out_edge;
    }
}

__global__
void device_print_edge_counter( uint32_t edge_ctr_in, uint32_t* d_edge_counter )
{
    if( edge_ctr_in != 0 ) {
        printf("    Function called with %d edges\n", edge_ctr_in );
    }
    printf("    Device sees edge counter %d\n", *d_edge_counter);
}

#if 0
__global__
void debug_gauss( cv::cuda::PtrStepSzf src )
{
    size_t non_null_ct = 0;
    float minval = 1000.0f;
    float maxval = -1000.0f;
    for( size_t i=0; i<src.rows; i++ )
        for( size_t j=0; j<src.cols; j++ ) {
            float f = src.ptr(i)[j];
            if( f != 0.0f )
                non_null_ct++;
            minval = min( minval, f );
            maxval = max( maxval, f );
        }
    printf("There are %d non-null values in the Gaussian end result (min %f, max %f)\n", non_null_ct, minval, maxval );
}
#endif

__host__
void Frame::initThinningTable( )
{
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut,
                                         h_thinning_lut,
                                         256*sizeof(unsigned char) );
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut_t,
                                         h_thinning_lut_t,
                                         256*sizeof(unsigned char) );
}

__host__
void Frame::applyMag( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    dim3 big_block;
    dim3 big_grid;
    big_block.x = 32;
    big_block.y = 32;
    big_grid.x  = ( getWidth()  / 32 ) + ( getWidth()  % 32 == 0 ? 0 : 1 );
    big_grid.y  = ( getHeight() / 32 ) + ( getHeight() % 32 == 0 ? 0 : 1 );

    // necessary to merge into 1 stream
    compute_mag_l2
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag );
    POP_CHK_CALL_IFSYNC;

    compute_map
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag, _d_map, 256.0f * params._cannyThrLow, 256.0f * params._cannyThrHigh );
    POP_CHK_CALL_IFSYNC;

    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void Frame::applyMore( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    thinning
        <<<grid,block,0,_stream>>>
        ( _d_hyst_edges, cv::cuda::PtrStepSzb(_d_intermediate) );
    POP_CHK_CALL_IFSYNC;

    const uint32_t max_num_edges = params._maxEdges;

    {
        uint32_t dummy = 0;
        POP_CUDA_MEMCPY_ASYNC( _d_edge_counter, &dummy, sizeof(uint32_t), cudaMemcpyHostToDevice, _stream );
    }

    thinning_and_store
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), _d_edges, _d_edge_counter, max_num_edges, _d_edgelist );
    POP_CHK_CALL_IFSYNC;

    POP_CUDA_MEMCPY_ASYNC( &_h_edgelist_sz, _d_edge_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, _stream );

    // Note: right here, Dynamic Parallelism would avoid blocking.
    cudaStreamSynchronize( _stream );
    // Try to figure out how that can be done with CMake.

    if( _h_edgelist_sz == 0 ) {
        cerr << "I have not found any edges!" << endl;
        cerr << "Leave " << __FUNCTION__ << endl;
        return;
    }

    const uint32_t nmax          = params._distSearch;
    const int32_t  threshold     = params._thrGradientMagInVote;
    block.x = 32;
    block.y = 2;
    block.z = 1;
    grid.x  = _h_edgelist_sz / 32 + ( _h_edgelist_sz % 32 != 0 ? 1 : 0 );
    grid.y  = 1;
    grid.z  = 1;

    {
        /* Note: the initial _d_edge_counter is set to 1 because it is used
         * as an index for writing points into a array. Starting the counter
         * at 1 allows to distinguish unchained points (0) from chained
         * points non-0.
         */
        uint32_t dummy = 1;
        POP_CUDA_MEMCPY_ASYNC( _d_edge_counter, &dummy, sizeof(uint32_t), cudaMemcpyHostToDevice, _stream );
    }

    cout << "    calling gradient descent with " << _h_edgelist_sz << " edge points" << endl;
    cout << "    max num edges is " << max_num_edges << endl;

    device_print_edge_counter
        <<<1,1,0,_stream>>>
        ( 0, _d_edge_counter );

    cout << "    grid (" << grid.x << "," << grid.y << "," << grid.z << ")"
         << " block (" << block.x << "," << block.y << "," << block.z << ")" << endl;

    gradient_descent
        <<<grid,block,0,_stream>>>
        ( _d_edgelist,   _h_edgelist_sz,
          _d_edgelist_2, _d_edge_counter,
          _d_next_edge_coord,
          // _d_next_edge_after,
          // _d_next_edge_befor,
          max_num_edges, _d_edges, nmax, _d_dx, _d_dy, threshold ); // , _d_out_points );
    POP_CHK_CALL_IFSYNC;

    device_print_edge_counter
        <<<1,1,0,_stream>>>
        ( _h_edgelist_sz, _d_edge_counter );

    POP_CUDA_MEMCPY_ASYNC( &_h_edgelist_2_sz, _d_edge_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost,_stream );

    device_print_edge_counter
        <<<1,1,0,_stream>>>
        ( 0, _d_edge_counter );

    // Note: right here, Dynamic Parallelism would avoid blocking.
    cudaStreamSynchronize( _stream );
    // Try to figure out how that can be done with CMake.

    cout << "    after gradient descent, edge counter is " << _h_edgelist_2_sz << endl;

    /*
     * After this:
     * - it is unclear whether a sweep over the entire edge plane
     *   is still an efficient choice, or whether it would be better
     *   to reduce the edge points into a flat array first.
     * - if reduction is the thing to do, it would be better to
     *   combine it with the second thinning step.
     *
     * cctagDetectionFromEdges takes as parameters:
     * - markers     : output, list of marker coordinates
     * - points      : a vector that contains all edge points (input)
     * - sourceView  : image at this scale (input)
     * - cannyGradX  : _d_dx (input)
     * - cannyGradY  : _d_dy (input)
     * - edgesMap    : _d_edges (input)
     * - frame       : running counter for video frames
     * - level       : this layer of the pyramid
     * - scale       : probably a coordinate multiplier
     * - params      : parameters
     *
     * calls vote
     *
     * vote takes as input:
     * - points     : a vector that contains all edge points (input)
     * - seeds      : unknown (output)
     * - edgesMap   : _d_edges (input)
     * - winners    : map of winners (output)
     * - cannyGradX : _d_dx (input)
     * - cannyGradY : _d_dy (input)
     * - params     : parameters
     *
     * calls gradientDirectionDescent
     *
     * gradientDirectionDescent takes as input
     * - canny       : _d_edges (input)
     * - p           : coordinate of one edge point (input)
     * - dir         : direction, 1 or -1
     * - nmax        : global parameter the distance from edge point for searching
     * - cannyGradX  : _d_dx
     * - cannyGradY  : _d_dy
     * - thrGradiant : global parameter for gradient thresholding
     * returns: NULL or a new point
     */
#if 0
    // very costly printf-debugging
    debug_gauss
        <<<1,1,0,_stream>>>
        ( _d_smooth );
#endif

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart

