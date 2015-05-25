#pragma once

#include <cuda_runtime.h>
#include <assert.h>

// #include <opencv2/core/core.hpp>
// #include <opencv2/core/core_c.h>
#include <opencv2/core/cuda_types.hpp>
// #include <opencv2/core/operations.hpp>
// #include <opencv2/imgproc/imgproc_c.h>
// #include <opencv2/imgproc/types_c.h>

namespace popart {

/*************************************************************
 * FrameTexture
 * Used by Frame to perform scaling with bilinear interpolation
 * and some other transformations.
 *************************************************************/
class FrameTexture
{
public:
    enum Kind {
        normalized_uchar_to_float
    };
public:
    FrameTexture( const cv::cuda::PtrStepSzb& plane );
    ~FrameTexture( );

    inline cudaTextureObject_t getTex( ) {
        return _texture;
    }

private:
    void makeTex_Normalized_uchar_to_float( const cv::cuda::PtrStepSzb& plane );

private:
    Kind                _kind;
    cudaTextureObject_t _texture;
    cudaTextureDesc     _texDesc;
    cudaResourceDesc    _resDesc;
};

/*************************************************************
 * Frame
 * The basic structure for managing data stored on the GPU
 *************************************************************/
class Frame
{
public:
    // create continuous device memory, enough for @layers copies of @width x @height
    Frame( uint32_t type_size, uint32_t width, uint32_t height );
    ~Frame( );

    static void initGaussTable( );

    // copy the upper layer from the host to the device
    void upload( const unsigned char* image ); // implicitly assumed that w/h are the same as above

    // copy a given number of layers from the device to the host
    void download( unsigned char* image, uint32_t width, uint32_t height );

    // Create a texture object this frame.
    // The caller must ensure that the Kind of texture object makes sense.
    void createTexture( FrameTexture::Kind kind );

    void deleteTexture( ); // Delete it. Done anyway in the destructor.

    // initialize this frame from other's normalized texture
    void fillFromTexture( Frame& src );
    void fillFromFrame( Frame& src );

    inline cudaTextureObject_t getTex( ) {
        assert( _texture );
        return _texture->getTex( );
    }

    void streamSync( ); // Wait for the asynchronous ops to finish

    // return the downscaled sibling "scale". The count it 0-based, 0 is this Frame
    Frame* getScale( uint32_t scale );

    // return width in type_size
    uint32_t getWidth( ) const  { return _d_plane.cols; }
    uint32_t getHeight( ) const { return _d_plane.rows; }
    uint32_t getPitch( ) const  { return _d_plane.step * _d_plane.elem_size; }

    void allocHostDebugPlane( );
    void allocDevGaussianPlane( );
    void applyGauss( );
    void hostDebugDownload( );
    static void writeDebugPlane( const char* filename, unsigned char* c, uint32_t w, uint32_t h );
    void writeHostDebugPlane( const char* filename );

private:
    Frame( );  // forbidden
    Frame( const Frame& );  // forbidden

private:
    uint32_t _type_size;
    cv::cuda::PtrStepSzb _d_plane;
    cv::cuda::PtrStepSzf _d_gaussian_intermediate;
    cv::cuda::PtrStepSzf _d_gaussian;

    uint32_t       _d_gaussian_pitch;
    unsigned char* _h_debug_plane;
    FrameTexture*  _texture;

    // if we run out of streams (there are 32), we may have to share
    bool         _stream_inherited;
    cudaStream_t _stream;
};

}; // namespace popart

