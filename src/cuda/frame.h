#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <string>

#include <opencv2/core/cuda_types.hpp>

#include "../cctag/params.hpp"

/* A table is copied to constant memory containing sigma values
 * for Gauss filtering at the 0-offset, and the derivatives
 * at +16.
 * Necessary to overcome alignment mess with __constant__
 * memory.
 */
#define GAUSS_TABLE  0 // Gauss parameters
#define GAUSS_DERIV 16 // first derivative

namespace cv {
    namespace cuda {
        typedef PtrStepSz<int16_t>  PtrStepSz16s;
        typedef PtrStepSz<uint32_t> PtrStepSz32u;
        typedef PtrStepSz<int32_t>  PtrStepSz32s;
        typedef PtrStep<int16_t>    PtrStep16s;
        typedef PtrStep<uint32_t>   PtrStep32u;
        typedef PtrStep<int32_t>    PtrStep32s;
    }
};

namespace popart {

typedef cudaEvent_t FrameEvent;

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
 * TriplePoint
 * A simplified version of EdgePoint in the C++ code.
 *************************************************************/
struct TriplePoint
{
    int2 coord;
    int2 befor;
    int2 after;
    int  next_coord; // I believe that this can be removed
    int  next_after;
    int  next_befor;
};

/*************************************************************
 * Frame
 * The basic structure for managing data stored on the GPU
 *************************************************************/
class Frame
{
public:
    // create continuous device memory, enough for @layers copies of @width x @height
    Frame( uint32_t width, uint32_t height );
    ~Frame( );

    // Copy manually created Gauss filter tables to constant memory
    // implemented in frame_gaussian.cu
    static void initGaussTable( );

    // Copy manually created LUT tables for thinning
    static void initThinningTable( );

    // copy the upper layer from the host to the device
    void upload( const unsigned char* image ); // implicitly assumed that w/h are the same as above

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

    void allocUploadEvent( );
    void deleteUploadEvent( );
    FrameEvent addUploadEvent( );
    void allocDoneEvent( );
    void deleteDoneEvent( );
    FrameEvent addDoneEvent( );
    void streamSync( ); // Wait for the asynchronous ops to finish
    void streamSync( FrameEvent ev ); // Wait for ev to happen (in another stream)

    // return the downscaled sibling "scale". The count it 0-based, 0 is this Frame
    Frame* getScale( uint32_t scale );

    // return width in type_size
    uint32_t getWidth( ) const  { return _d_plane.cols; }
    uint32_t getHeight( ) const { return _d_plane.rows; }
    uint32_t getPitch( ) const  { return _d_plane.step; }

    // implemented in frame_gaussian.cu
    void allocDevGaussianPlane( const cctag::Parameters& param );

    // implemented in frame_gaussian.cu
    void applyGauss( const cctag::Parameters& param );

    // implemented in frame_apply.cu
    void applyMag( const cctag::Parameters& param );

    // implemented in frame_hyst.cu
    void applyHyst( const cctag::Parameters& param );

    // implemented in frame_apply.cu
    void applyMore( const cctag::Parameters& param );

    void hostDebugDownload( const cctag::Parameters& params ); // async

    static void writeDebugPlane1( const char* filename, const cv::cuda::PtrStepSzb& plane );

    template<class T>
    static void writeDebugPlane( const char* filename, const cv::cuda::PtrStepSz<T>& plane );

    static void writeInt2Array( const char* filename, const int2* array, uint32_t sz );
    static void writeTriplePointArray( const char* filename, const TriplePoint* array, uint32_t sz );

    void writeHostDebugPlane( std::string filename, const cctag::Parameters& params );
    void hostDebugCompare( unsigned char* pix );

private:
    Frame( );  // forbidden
    Frame( const Frame& );  // forbidden

private:
    cv::cuda::PtrStepSzb _d_plane;
    cv::cuda::PtrStepSzf _d_intermediate;
    cv::cuda::PtrStepSzf _d_smooth;
    cv::cuda::PtrStepSz16s _d_dx; // cv::cuda::PtrStepSzf _d_dx;
    cv::cuda::PtrStepSz16s _d_dy; // cv::cuda::PtrStepSzf _d_dy;
    cv::cuda::PtrStepSz32u _d_mag;
    cv::cuda::PtrStepSzb   _d_map;
    cv::cuda::PtrStepSzb   _d_hyst_edges;
    cv::cuda::PtrStepSzb   _d_edges;
    int2*                  _d_edgelist;
    TriplePoint*           _d_edgelist_2;
    uint32_t*              _d_edge_counter;
    cv::cuda::PtrStepSz32s _d_next_edge_coord; // 2D plane for chaining TriplePoint coord
    cv::cuda::PtrStepSz32s _d_next_edge_after; // 2D plane for chaining TriplePoint after
    cv::cuda::PtrStepSz32s _d_next_edge_befor; // 2D plane for chaining TriplePoint after

    unsigned char* _h_debug_plane;
    float*         _h_debug_smooth;
    int16_t*       _h_debug_dx;
    int16_t*       _h_debug_dy;
    uint32_t*      _h_debug_mag;
    unsigned char* _h_debug_map;
    unsigned char* _h_debug_hyst_edges;
    unsigned char* _h_debug_edges;
    int2*          _h_debug_edgelist;
    uint32_t       _h_edgelist_sz;
    TriplePoint*   _h_debug_edgelist_2;
    uint32_t       _h_edgelist_2_sz;

    FrameTexture*  _texture;
    FrameEvent*    _wait_for_upload;
    FrameEvent*    _wait_done;

    // if we run out of streams (there are 32), we may have to share
    // bool         _stream_inherited;
public:
    cudaStream_t _stream;
};

}; // namespace popart

