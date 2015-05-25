#include <iostream>
#include <fstream>
#include <string.h>
#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "frame_gaussian.h"

namespace popart {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

Frame::Frame( uint32_t type_size, uint32_t width, uint32_t height )
    : _type_size( type_size )
    , _width( width )
    , _height( height )
    , _d_gaussian_intermediate( 0 )
    , _d_gaussian( 0 )
    , _d_gaussian_pitch( 0 )
    , _h_debug_plane( 0 )
    , _texture( 0 )
    , _stream_inherited( false )
{
    cerr << "Allocating frame: " << width << "x" << height << " (typesize " << type_size << ")" << endl;

    POP_CUDA_STREAM_CREATE( &_stream );

    size_t pitch;
    POP_CUDA_MALLOC_PITCH( (void**)&_d_plane, &pitch, width*type_size, height );
    _pitch = pitch;

    cudaError_t err;
    err = cudaMemsetAsync( _d_plane, 0, width*type_size*height );
    POP_CUDA_FATAL_TEST( err, "cudaMemsetAsync failed: " );
}

Frame::~Frame( )
{
    delete _h_debug_plane;
    delete _texture;

    POP_CUDA_FREE( _d_gaussian_intermediate );
    POP_CUDA_FREE( _d_gaussian );
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

void Frame::allocDevGaussianPlane( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    size_t pitch;
    POP_CUDA_MALLOC_PITCH( (void**)&_d_gaussian, &pitch, _width*sizeof(float), _height );
    _d_gaussian_pitch = pitch;

    POP_CUDA_MALLOC_PITCH( (void**)&_d_gaussian_intermediate, &pitch, _width*sizeof(float), _height );

    cerr << "Leave " << __FUNCTION__ << endl;
}

void Frame::applyGauss( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = V7_WIDTH;
    grid.x  = _width / V7_WIDTH;
    grid.y  = _height;

    // the kernels take float* and measure width in steps of float
    const uint32_t element_pitch = _d_gaussian_pitch / sizeof(float);

    filter_gauss_horiz_from_uchar
        <<<grid,block,0,_stream>>>
        ( _d_plane,
          _d_gaussian_intermediate,
          _width, _pitch, _height );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate,
          _d_gaussian,
          _width, element_pitch, _height );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian,
          _d_gaussian_intermediate,
          _width, element_pitch, _height );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate,
          _d_gaussian,
          _width, element_pitch, _height );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian,
          _d_gaussian_intermediate,
          _width, element_pitch, _height );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate,
          _d_gaussian,
          _width, element_pitch, _height );

    cerr << "Leave " << __FUNCTION__ << endl;
}

void Frame::allocHostDebugPlane( )
{
    delete [] _h_debug_plane;
    _h_debug_plane = new unsigned char[ _type_size * _width * _height ];
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

void Frame::hostDebugDownload( )
{
    allocHostDebugPlane( );
    download( _h_debug_plane, _width, _height );
}

void Frame::writeDebugPlane( const char* filename, unsigned char* c, uint32_t w, uint32_t h )
{
    ofstream of( filename );
    of << "P5" << endl
       << w << " " << h << endl
       << "255" << endl;
    of.write( (char*)c, w * h );
}

void Frame::writeHostDebugPlane( const char* filename )
{
    assert( _h_debug_plane );

    if( _type_size == 1 ) {
        ofstream of( filename );
        of << "P5" << endl
           << _width << " " << _height << endl
           << "255" << endl;
        of.write( (char*)_h_debug_plane, _width * _height );
    } else if( _type_size == 4 ) {
        ofstream of( filename );
        of << "P5" << endl
           << _width << " " << _height << endl
           << "255" << endl;
        for( uint32_t h=0; h<_height; h++ ) {
            for( uint32_t w=0; w<_width; w++ ) {
                of << (unsigned char)_h_debug_plane[h*_width+w];
            }
        }
    } else {
        cerr << "Only type_sizes 1 (uint8_t) and 4 (non-normalized flaot) are supported" << endl;
    }
}

void Frame::createTexture( FrameTexture::Kind kind )
{
    if( _texture ) delete _texture;

    assert( _type_size == 1 );
    _texture = new FrameTexture( FrameTexture::normalized_uchar_to_float, _d_plane, _pitch, _width, _height );
}

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
    cerr << "Entering " << __FUNCTION__ << endl;
    cerr << "    copying from src frame with " << src._width << "x" << src._height << endl;
    cerr << "    to dst plane           with " << _width << "x" << _height << endl;
    assert( _d_plane );
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = _width / 32;
    grid.y  = _height;

    cu_fill_from_frame
        <<<grid,block,0,_stream>>>
        ( _d_plane, _pitch, _width, _height, src._d_plane, src._pitch, src._width, src._height );
}

__global__
void cu_fill_from_texture( unsigned char* dst, uint32_t pitch, uint32_t width, uint32_t height, cudaTextureObject_t tex )
{
    uint32_t idy = blockIdx.y;
    uint32_t idx = blockIdx.x * 32 + threadIdx.x;
    if( idy >= height ) return;
    if( idx >= pitch ) return;
    float d = tex2D<float>( tex, float(idx)/float(width), float(idy)/float(height) );
    dst[ idy * pitch + idx ] = (unsigned char)( d * 255 );
}

void Frame::fillFromTexture( Frame& src )
{
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = _width / 32;
    grid.y  = _height;

    cu_fill_from_texture
        <<<grid,block,0,_stream>>>
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

