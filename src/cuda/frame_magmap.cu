#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"


namespace popart
{

using namespace std;

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
    // --- rintf( hypot ( ) ) --
    dx *= dx;
    dy *= dy;
    dst.ptr(idy)[idx] = __fsqrt_rn( (float)( dx + dy ) );
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

    assert( edge_type <= 2 );
    map.ptr(idy)[idx] = edge_type;
}

__host__
void Frame::applyMag( const cctag::Parameters & params )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

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

    /* block download until MAG is ready */
    cudaEventRecord( &_download_ready_event.mag, _stream );
    cudaStreamWaitEvent( _download_stream, _download_ready_event.mag );

    POP_CUDA_MEMCPY_2D_ASYNC( _h_mag.data, _h_mag.step,
                              _d_mag.data, _d_mag.step,
                              _d_mag.cols * sizeof(uint32_t),
                              _d_mag.rows,
                              cudaMemcpyDeviceToHost, _download_stream );

    compute_map
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag, _d_map, 256.0f * params._cannyThrLow, 256.0f * params._cannyThrHigh );
    POP_CHK_CALL_IFSYNC;

    /* block download until MAG is ready */
    cudaEventRecord( &_download_ready_event.map, _stream );
    cudaStreamWaitEvent( _download_stream, _download_ready_event.map );

#ifdef DEBUG_WRITE_MAP_AS_PGM
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_map, getWidth() * sizeof(uint8_t),
                              _d_map.data, _d_map.step,
                              _d_map.cols * sizeof(uint8_t),
                              _d_map.rows,
                              cudaMemcpyDeviceToHost, _download_stream );
#endif // DEBUG_WRITE_MAP_AS_PGM

    // cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart

