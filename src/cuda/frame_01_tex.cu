#include "frame.h"
#include "debug_macros.hpp"

#include <cuda_runtime.h>

namespace popart {

using namespace std;

__global__
void cu_fill_from_texture( cv::cuda::PtrStepSzb dst, cudaTextureObject_t tex )
{
    uint32_t idy = blockIdx.y;
    uint32_t idx = blockIdx.x * 32 + threadIdx.x;
    if( idy >= dst.rows ) return;
    if( idx >= dst.step ) return;
    bool nix = ( idx < dst.cols );
    float d = tex2D<float>( tex, float(idx)/float(dst.cols), float(idy)/float(dst.rows) );
    dst.ptr(idy)[idx] = nix ? (unsigned char)( d * 255 ) : 0;
}

void Frame::fillFromTexture( Frame& src )
{
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    cu_fill_from_texture
        <<<grid,block,0,_stream>>>
        ( _d_plane, src.getTex() );
    POP_CHK_CALL_IFSYNC;
}

__host__
void Frame::applyPlaneDownload( )
{
    cudaEventRecord( _download_ready_event.plane, _stream );

    cudaStreamWaitEvent( _download_stream, _download_ready_event.plane, 0 );

    // download - layer 0 is mandatory, other layers for debugging
    cudaMemcpy2DAsync( _h_plane.data, _h_plane.step,
                       _d_plane.data, _d_plane.step,
                       _d_plane.cols,
                       _d_plane.rows,
                       cudaMemcpyDeviceToHost, _download_stream );
}

}; // namespace popart

