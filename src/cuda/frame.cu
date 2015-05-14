#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"

namespace popart {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

Frame::Frame( uint32_t type_size, uint32_t width, uint32_t height )
    : _type_size( type_size )
    , _width( width )
    , _height( height )
    , _texture( 0 )
    , _stream_inherited( false )
{
    cerr << "Allocating frame: " << width << "x" << height << " (typesize " << type_size << ")" << endl;

    POP_CUDA_STREAM_CREATE( &_stream );

    size_t pitch;
    POP_CUDA_MALLOC_PITCH( (void**)&_d_plane, &pitch, width*type_size, height );
    _pitch = pitch;
}

Frame::~Frame( )
{
    delete _texture;

    POP_CUDA_FREE( _d_plane );
    cerr << "Released frame: " << _width << "x" << _height << endl;
}

void Frame::upload( const unsigned char* image )
{
    const uint32_t w = _width * _type_size;

    cerr << "source w=" << w
         << " source pitch=" << w
         << " dest pitch=" << _pitch
         << " height=" << _height << endl;
    POP_CUDA_MEMCPY_2D_ASYNC( _d_plane, _pitch, image, w, w, _height, cudaMemcpyHostToDevice, _stream );
}

void Frame::download( unsigned char* image, uint32_t width, uint32_t height )
{
    const uint32_t host_w = width * _type_size;
    const uint32_t dev_w  = _width * _type_size;
    assert( host_w >= _width );
    assert( height >= _height );
    cerr << "source dev_w=" << dev_w
         << " source pitch=" << _pitch
         << " dest pitch=" << host_w
         << " height=" << _height << endl;
    POP_CUDA_MEMCPY_2D_ASYNC( image, host_w, _d_plane, _pitch, dev_w, _height, cudaMemcpyDeviceToHost, _stream );
}

void Frame::createTexture( FrameTexture::Kind kind )
{
    if( _texture ) delete _texture;

    assert( _type_size == 1 );
    _texture = new FrameTexture( FrameTexture::normalized_uchar_to_float, _d_plane, _pitch, _width, _height );
}

__global__
void cu_fill_from_texture( unsigned char* dst, uint32_t pitch, uint32_t width, uint32_t height, cudaTextureObject_t tex )
{
    uint32_t idy = blockIdx.y;
    uint32_t idx = blockIdx.x * 32 + threadIdx.x;
    if( idy >= height ) return;
    if( idx >= pitch ) return;
    float d = tex2D<float>( tex, float(idx)/float(width), float(idy)/float(height) );
    dst[ idy * pitch + idx ] = (unsigned char)( d * 256 );
}

void Frame::fillFromTexture( Frame& src )
{
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = _width / 32;
    grid.y  = _height;

    cu_fill_from_texture
        <<<block,grid,0,_stream>>>
        ( _d_plane, _pitch, _width, _height, src.getTex() );
}

void Frame::deleteTexture( )
{
    delete _texture;
    _texture = 0;
}

void Frame::streamSync( )
{
    cudaStreamSynchronize( _stream );
}

/*************************************************************
 * FrameTexture
 *************************************************************/

void FrameTexture::makeTex_Normalized_uchar_to_float( void* ptr, uint32_t pitch, uint32_t width, uint32_t height )
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
    _resDesc.res.pitch2D.devPtr       = ptr;
    _resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    _resDesc.res.pitch2D.desc.x       = 8;
    _resDesc.res.pitch2D.desc.y       = 0;
    _resDesc.res.pitch2D.desc.z       = 0;
    _resDesc.res.pitch2D.desc.w       = 0;
    _resDesc.res.pitch2D.pitchInBytes = pitch;
    _resDesc.res.pitch2D.width        = width;
    _resDesc.res.pitch2D.height       = height;

    cudaError_t err;
    err = cudaCreateTextureObject( &_texture, &_resDesc, &_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

FrameTexture::FrameTexture( Kind k, void* ptr, uint32_t pitch, uint32_t w, uint32_t h )
    : _kind( k )
{
    switch( k )
    {
    case normalized_uchar_to_float :
        makeTex_Normalized_uchar_to_float( ptr, pitch, w, h );
        break;
    default :
        cerr << "Unsupported texture creation mode" << endl;
        exit( -__LINE__ );
        break;
    }
}

FrameTexture::~FrameTexture( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _texture );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
}

}; // namespace popart

