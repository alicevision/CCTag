#include <iostream>
#include <limits>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "cctag/talk.hpp"
// #include "clamp.h"
// #include "frame_gaussian.h"

namespace popart {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

Frame::Frame( uint32_t width, uint32_t height, int my_layer, cudaStream_t download_stream, int my_pipe )
    : _layer( my_layer )
    , _h_debug_hyst_edges( 0 )
    , _texture( 0 )
    , _wait_for_upload( 0 )
    , _meta( my_pipe, my_layer )
{
    DO_TALK( cerr << "Allocating frame: " << width << "x" << height << endl; )
#ifndef EDGE_LINKING_HOST_SIDE
    _h_ring_output.data = 0;
#endif

    if( download_stream != 0 ) {
        _private_download_stream = false;
        _download_stream = download_stream;
    } else {
        _private_download_stream = true;
        cudaStreamCreateWithFlags( &_download_stream, cudaStreamNonBlocking );
        // POP_CUDA_STREAM_CREATE( &_download_stream );
    }
    POP_CUDA_STREAM_CREATE( &_stream );

    // POP_CUDA_EVENT_CREATE( &_download_ready_event );
    // at least in older CUDA versions, events blocked parallelism
    cudaEventCreateWithFlags( &_stream_done,                      cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_stream_done,             cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.plane,       cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.dxdy,        cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.magmap,      cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.edgecoords1, cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.edgecoords2, cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.descent1,    cudaEventDisableTiming);
    cudaEventCreateWithFlags( &_download_ready_event.descent2,    cudaEventDisableTiming);

    size_t pitch;
    POP_CUDA_MALLOC_PITCH( (void**)&_d_plane.data, &pitch, width, height );
    _d_plane.step = pitch;
    _d_plane.cols = width;
    _d_plane.rows = height;
    assert( pitch % _d_plane.elemSize() == 0 );

    POP_CUDA_MEMSET_ASYNC( _d_plane.data,
                           0,
                           _d_plane.step * _d_plane.rows,
                           _stream );
}

Frame::~Frame( )
{
    deleteUploadEvent( );

    releaseRequiredMem( );

    // host-side plane for debugging
    delete [] _h_debug_hyst_edges;

    // required host-side planes
    delete _texture;

    cudaEventDestroy( _stream_done );
    cudaEventDestroy( _download_stream_done );
    cudaEventDestroy( _download_ready_event.plane );
    cudaEventDestroy( _download_ready_event.dxdy );
    cudaEventDestroy( _download_ready_event.magmap );
    cudaEventDestroy( _download_ready_event.edgecoords1 );
    cudaEventDestroy( _download_ready_event.edgecoords2 );
    cudaEventDestroy( _download_ready_event.descent1 );
    cudaEventDestroy( _download_ready_event.descent2 );

    if( _private_download_stream ) {
        POP_CUDA_STREAM_DESTROY( _download_stream );
    }
    POP_CUDA_STREAM_DESTROY( _stream );
}

void Frame::upload( const unsigned char* image )
{
    DO_TALK(
      cerr << "source w=" << _d_plane.cols
           << " source pitch=" << _d_plane.cols
           << " dest pitch=" << _d_plane.step
           << " height=" << _d_plane.rows
           << endl;)
    POP_CUDA_MEMCPY_2D_ASYNC( _d_plane.data,
                              getPitch(),
                              image,
                              getWidth(),
                              getWidth(),
                              getHeight(),
                              cudaMemcpyHostToDevice, _stream );
}

void Frame::createTexture( FrameTexture::Kind kind )
{
    if( _texture ) delete _texture;

    _texture = new FrameTexture( _d_plane );
}

#if 0
__global__
void cu_fill_from_frame( unsigned char* dst, uint32_t pitch, uint32_t width, uint32_t height, unsigned char* src, uint32_t spitch, uint32_t swidth, uint32_t sheight )
{
    uint32_t idy = blockIdx.y;
    uint32_t idx = blockIdx.x * 32 + threadIdx.x;
    if( idy >= height ) return;
    if( idx >= pitch ) return;

    dst[ idy * pitch + idx ] = src[ idy * spitch + idx ];
}

void Frame::fillFromFrame( Frame& src )
{
    DO_TALK(
      cerr << "Entering " << __FUNCTION__ << endl;
      cerr << "    copying from src frame with " << src.getWidth() << "x" << src.getHeight() << endl;
      cerr << "    to dst plane           with " << getWidth() << "x" << getHeight() << endl;
    )
    assert( _d_plane );
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = getWidth() / 32;
    grid.y  = getHeight();

    cu_fill_from_frame
        <<<grid,block,0,_stream>>>
        ( _d_plane, getPitch(), getWidth(), getHeight(), src._d_plane, src.getPitch(), src.getWidth(), src.getHeight() );
    POP_CHK_CALL_IFSYNC;
}
#endif

void Frame::deleteTexture( )
{
    delete _texture;
    _texture = 0;
}

void Frame::allocUploadEvent( )
{
    _wait_for_upload = new cudaEvent_t;

    cudaError_t err;
    err = cudaEventCreateWithFlags( _wait_for_upload, cudaEventDisableTiming );
    POP_CUDA_FATAL_TEST( err, "Could not create a non-timing event: " );
}

void Frame::deleteUploadEvent( )
{
    if( not _wait_for_upload ) return;
    cudaEventDestroy( *_wait_for_upload );
    delete _wait_for_upload;
}

cudaEvent_t Frame::addUploadEvent( )
{
    cudaError_t err;
    err = cudaEventRecord( *_wait_for_upload, _stream );
    POP_CUDA_FATAL_TEST( err, "Could not insert an event into a stream: " );
    return *_wait_for_upload;
}

void Frame::streamSync( )
{
    cudaStreamSynchronize( _stream );
}

void Frame::streamSync( cudaEvent_t ev )
{
    cudaStreamWaitEvent( _stream, ev, 0 );
}

/*************************************************************
 * FrameTexture
 *************************************************************/

void FrameTexture::makeTex_Normalized_uchar_to_float( const cv::cuda::PtrStepSzb& plane )
{
    memset( &_texDesc, 0, sizeof(cudaTextureDesc) );

    _texDesc.normalizedCoords = 1;                           // address 0..1 instead of 0..width/height
    _texDesc.addressMode[0]   = cudaAddressModeClamp;
    _texDesc.addressMode[1]   = cudaAddressModeClamp;
    _texDesc.addressMode[2]   = cudaAddressModeClamp;
    _texDesc.readMode         = cudaReadModeNormalizedFloat; // automatic conversion from uchar to float
    _texDesc.filterMode       = cudaFilterModeLinear;        // bilinear interpolation

    memset( &_resDesc, 0, sizeof(cudaResourceDesc) );
    _resDesc.resType                  = cudaResourceTypePitch2D;
    _resDesc.res.pitch2D.devPtr       = plane.data;
    _resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    _resDesc.res.pitch2D.desc.x       = 8;
    _resDesc.res.pitch2D.desc.y       = 0;
    _resDesc.res.pitch2D.desc.z       = 0;
    _resDesc.res.pitch2D.desc.w       = 0;
    assert( plane.elemSize() == 1 );
    _resDesc.res.pitch2D.pitchInBytes = plane.step;
    _resDesc.res.pitch2D.width        = plane.cols;
    _resDesc.res.pitch2D.height       = plane.rows;

    cudaError_t err;
    err = cudaCreateTextureObject( &_texture, &_resDesc, &_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

FrameTexture::FrameTexture( const cv::cuda::PtrStepSzb& plane )
    : _kind( normalized_uchar_to_float )
{
    makeTex_Normalized_uchar_to_float( plane );
}

FrameTexture::~FrameTexture( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _texture );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
}

}; // namespace popart

