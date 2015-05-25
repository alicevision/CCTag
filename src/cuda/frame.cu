#include <iostream>
#include <assert.h>
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
    , _d_gaussian_pitch( 0 )
    , _h_debug_plane( 0 )
    , _texture( 0 )
    , _stream_inherited( false )
{
    cerr << "Allocating frame: " << width << "x" << height << " (typesize " << type_size << ")" << endl;

    POP_CUDA_STREAM_CREATE( &_stream );

    size_t pitch;
    POP_CUDA_MALLOC_PITCH( (void**)&_d_plane.data, &pitch, width*type_size, height );
    _d_plane.step = pitch / _d_plane.elemSize();
    _d_plane.cols = width;
    _d_plane.rows = height;
    assert( pitch % _d_plane.elemSize() == 0 );

    POP_CUDA_MEMSET_ASYNC( _d_plane.data,
                           0,
                           _d_plane.step * _d_plane.elemSize() * _d_plane.rows,
                           _stream );
}

Frame::~Frame( )
{
    delete _h_debug_plane;
    delete _texture;

    POP_CUDA_FREE( _d_plane.data );
    POP_CUDA_FREE( _d_gaussian_intermediate.data );
    POP_CUDA_FREE( _d_gaussian.data );
    cerr << "Released frame: " << getWidth() << "x" << getHeight() << endl;
}

void Frame::upload( const unsigned char* image )
{
    cerr << "source w=" << _d_plane.cols
         << " source pitch=" << _d_plane.cols
         << " dest pitch=" << _d_plane.step * _d_plane.elemSize()
         << " height=" << _d_plane.rows
         << endl;
    POP_CUDA_MEMCPY_2D_ASYNC( _d_plane.data,
                              getPitch(),
                              image,
                              getWidth(),
                              getWidth(),
                              getHeight(),
                              cudaMemcpyHostToDevice, _stream );
}

void Frame::allocDevGaussianPlane( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    void* ptr;
    const size_t w = getWidth();
    const size_t h = getHeight();
    size_t p;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    assert( p % _d_gaussian.elemSize() == 0 );
    _d_gaussian.data = (float*)ptr;
    _d_gaussian.step = p / _d_gaussian.elemSize();
    _d_gaussian.cols = w;
    _d_gaussian.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    _d_gaussian_intermediate.data = (float*)ptr;
    _d_gaussian_intermediate.step = p / _d_gaussian_intermediate.elemSize();
    _d_gaussian_intermediate.cols = w;
    _d_gaussian_intermediate.rows = h;

    cerr << "Leave " << __FUNCTION__ << endl;
}

void Frame::applyGauss( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = V7_WIDTH;
    grid.x  = getWidth() / V7_WIDTH;
    grid.y  = getHeight();

    filter_gauss_horiz_from_uchar
        <<<grid,block,0,_stream>>>
        ( _d_plane, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    cerr << "Leave " << __FUNCTION__ << endl;
}

void Frame::allocHostDebugPlane( )
{
    delete [] _h_debug_plane;
    _h_debug_plane = new unsigned char[ _type_size * getWidth() * getHeight() ];
}

void Frame::download( unsigned char* image, uint32_t width, uint32_t height )
{
    const uint32_t host_w = width * _type_size;
    const uint32_t dev_w  = getWidth() * _type_size;
    assert( host_w >= getWidth() );
    assert( height >= getHeight() );
    cerr << "source dev_w=" << dev_w
         << " source pitch=" << getPitch()
         << " dest pitch=" << host_w
         << " height=" << getHeight() << endl;
    POP_CUDA_MEMCPY_2D_ASYNC( image, host_w, _d_plane, getPitch(), dev_w, getHeight(), cudaMemcpyDeviceToHost, _stream );
}

void Frame::hostDebugDownload( )
{
    allocHostDebugPlane( );
    download( _h_debug_plane, getWidth(), getHeight() );
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
           << getWidth() << " " << getHeight() << endl
           << "255" << endl;
        of.write( (char*)_h_debug_plane, getWidth() * getHeight() );
    } else if( _type_size == 4 ) {
        ofstream of( filename );
        of << "P5" << endl
           << getWidth() << " " << getHeight() << endl
           << "255" << endl;
        for( uint32_t h=0; h<getHeight(); h++ ) {
            for( uint32_t w=0; w<getWidth(); w++ ) {
                of << (unsigned char)_h_debug_plane[h*getWidth()+w];
            }
        }
    } else {
        cerr << "Only type_sizes 1 (uint8_t) and 4 (non-normalized float) are supported" << endl;
    }
}

void Frame::createTexture( FrameTexture::Kind kind )
{
    if( _texture ) delete _texture;

    assert( _type_size == 1 );
    _texture = new FrameTexture( _d_plane );
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
    cerr << "    copying from src frame with " << src.getWidth() << "x" << src.getHeight() << endl;
    cerr << "    to dst plane           with " << getWidth() << "x" << getHeight() << endl;
    assert( _d_plane );
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = getWidth() / 32;
    grid.y  = getHeight();

    cu_fill_from_frame
        <<<grid,block,0,_stream>>>
        ( _d_plane, getPitch(), getWidth(), getHeight(), src._d_plane, src.getPitch(), src.getWidth(), src.getHeight() );
}

__global__
// void cu_fill_from_texture( unsigned char* dst, uint32_t pitch, uint32_t width, uint32_t height, cudaTextureObject_t tex )
void cu_fill_from_texture( cv::cuda::PtrStepSzb dst, cudaTextureObject_t tex )
{
    uint32_t idy = blockIdx.y;
    uint32_t idx = blockIdx.x * 32 + threadIdx.x;
    if( idy >= dst.rows ) return;
    if( idx >= dst.step ) return;
    bool nix = ( idx < dst.cols );
    float d = tex2D<float>( tex, float(idx)/float(dst.cols), float(idy)/float(dst.rows) );
    dst.ptr(idy)[idx] = nix ? (unsigned char)( d * 255 ) : 0;
    // dst[ idy * dst.step + idx ] = nix ? (unsigned char)( d * 255 ) : 0;
}

void Frame::fillFromTexture( Frame& src )
{
    dim3 grid;
    dim3 block;
    block.x = 32;
    grid.x  = getWidth() / 32;
    grid.y  = getHeight();

    cu_fill_from_texture
        <<<grid,block,0,_stream>>>
        // ( _d_plane, getPitch(), getWidth(), getHeight(), src.getTex() );
        ( _d_plane, src.getTex() );
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
    _resDesc.res.pitch2D.pitchInBytes = plane.step; // * plane.elemSize()
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

