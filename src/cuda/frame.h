#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

// #include <opencv2/core/cuda_types.hpp>

#include "cctag/params.hpp"
#include "cctag/types.hpp"
#include "frame_vote.h"
#include "triple_point.h"

#define DEBUG_WRITE_ORIGINAL_AS_PGM
#define DEBUG_WRITE_ORIGINAL_AS_ASCII
#undef  DEBUG_WRITE_GAUSSIAN_AS_PGM   // no longer computed
#undef  DEBUG_WRITE_GAUSSIAN_AS_ASCII // no longer computed
#define DEBUG_WRITE_DX_AS_PGM
#define DEBUG_WRITE_DX_AS_ASCII
#define DEBUG_WRITE_DY_AS_PGM
#define DEBUG_WRITE_DY_AS_ASCII
#define DEBUG_WRITE_MAG_AS_PGM
#define DEBUG_WRITE_MAG_AS_ASCII
#define DEBUG_WRITE_MAP_AS_PGM
#define DEBUG_WRITE_MAP_AS_ASCII
#define DEBUG_WRITE_HYSTEDGES_AS_PGM
#define DEBUG_WRITE_EDGES_AS_PGM
#define DEBUG_WRITE_EDGELIST_AS_PPM
#define DEBUG_WRITE_EDGELIST_AS_ASCII
#define DEBUG_WRITE_VOTERS_AS_PPM
#define DEBUG_WRITE_CHOSEN_AS_PPM
#define DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII
#define DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII
#define DEBUG_WRITE_LINKED_AS_PPM
#define DEBUG_WRITE_LINKED_AS_PPM_INTENSE
#define DEBUG_WRITE_LINKED_AS_ASCII
#define DEBUG_WRITE_LINKED_AS_ASCII_INTENSE

#define DEBUG_LINKED_USE_INT4_BUFFER

#define RESERVE_MEM_MAX_CROWNS  5

#define EDGE_LINKING_HOST_SIDE

#define EDGE_LINKING_MAX_EDGE_LENGTH        100
#define EDGE_LINKING_MAX_ARCS             10000
#define EDGE_LINKING_MAX_RING_BUFFER_SIZE    40

/* Separable compilation allows one kernel to instantiate
 * others. That avoids complexity on the host side when,
 * e.g., GPU-side counters need to be checked before starting
 * a new kernel.
 */
#define USE_SEPARABLE_COMPILATION

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
        typedef PtrStepSz<uchar4>   PtrStepSzb4;

        typedef PtrStep<int16_t>    PtrStep16s;
        typedef PtrStep<uint32_t>   PtrStep32u;
        typedef PtrStep<int32_t>    PtrStep32s;
        typedef PtrStep<uchar4>     PtrStepb4;

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
        typedef PtrStepSz<int4>     PtrStepSzInt2;
        typedef PtrStep<int4>       PtrStepInt2;
        typedef int4                PtrStepInt2_base_t;
#else // DEBUG_LINKED_USE_INT4_BUFFER
        typedef PtrStepSz<int2>     PtrStepSzInt2;
        typedef PtrStep<int2>       PtrStepInt2;
        typedef int2                PtrStepInt2_base_t;
#endif // DEBUG_LINKED_USE_INT4_BUFFER
    }
};


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
    Frame( uint32_t width, uint32_t height, int my_layer );
    ~Frame( );

public:
    int getLayer() const { return _layer; }

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
    cudaEvent_t addUploadEvent( );

    void streamSync( ); // Wait for the asynchronous ops to finish
    void streamSync( cudaEvent_t ev ); // Wait for ev to happen (in another stream)

    // return the downscaled sibling "scale". The count it 0-based, 0 is this Frame
    Frame* getScale( uint32_t scale );

    // return width in type_size
    uint32_t getWidth( ) const  { return _d_plane.cols; }
    uint32_t getHeight( ) const { return _d_plane.rows; }
    uint32_t getPitch( ) const  { return _d_plane.step; }

    // implemented in frame_alloc.cu
    void allocRequiredMem( const cctag::Parameters& param );
    void initRequiredMem( );
    void releaseRequiredMem( );

    // implemented in frame_gaussian.cu
    void applyGauss( const cctag::Parameters& param );

    // implemented in frame_gaussian.cu
    void applyGaussDownload( const cctag::Parameters& param );

    // implemented in frame_magmap.cu
    void applyMag( const cctag::Parameters& param );

    // implemented in frame_magmap.cu
    void applyMagDownload( const cctag::Parameters& param );

    // implemented in frame_hyst.cu
    void applyHyst( const cctag::Parameters& param );

    // implemented in frame_thin.cu
    void applyThinning( const cctag::Parameters& param );

    // implemented in frame_thin.cu
    void applyThinDownload( const cctag::Parameters& param );

    // implemented in frame_graddesc.cu
    bool applyDesc( const cctag::Parameters& param );

    // implemented in frame_graddesc.cu
    void applyDescDownload( const cctag::Parameters& param );

    // implemented in frame_vote.cu
    void applyVote( const cctag::Parameters& param );

    // implemented in frame_link.cu
    void applyLink( const cctag::Parameters& param );

    // implemented in frame_export.cu
    bool applyExport( std::vector<cctag::EdgePoint>&  vPoints,
                      cctag::EdgePointsImage&         edgesMap,
                      std::vector<cctag::EdgePoint*>& seeds,
                      cctag::WinnerMap&               winners );

    cv::Mat* getPlane( ) const;
    cv::Mat* getDx( ) const;
    cv::Mat* getDy( ) const;
    cv::Mat* getMag( ) const;
    cv::Mat* getEdges( ) const;

    void hostDebugDownload( const cctag::Parameters& params ); // async

    static void writeInt2Array( const char* filename, const int2* array, uint32_t sz );
    static void writeTriplePointArray( const char* filename, const TriplePoint* array, uint32_t sz );

    void writeHostDebugPlane( std::string filename, const cctag::Parameters& params );

    void hostDebugCompare( unsigned char* pix );

private:
    Frame( );  // forbidden
    Frame( const Frame& );  // forbidden
    Frame& operator=( const Frame& ); // forbidden

private:
    int                     _layer;

    int*                    _d_hysteresis_block_counter;
    int*                    _d_connect_component_block_counter;
    int*                    _d_ring_counter;
    int                     _d_ring_counter_max;

    cv::cuda::PtrStepSzb    _d_plane;
    cv::cuda::PtrStepSzf    _d_intermediate;
    cv::cuda::PtrStepSzf    _d_smooth;
    cv::cuda::PtrStepSz16s  _d_dx; // cv::cuda::PtrStepSzf _d_dx;
    cv::cuda::PtrStepSz16s  _d_dy; // cv::cuda::PtrStepSzf _d_dy;
    cv::cuda::PtrStepSz32u  _d_mag;
    cv::cuda::PtrStepSzb    _d_map;
    cv::cuda::PtrStepSzb    _d_hyst_edges;
    cv::cuda::PtrStepSzb    _d_edges;
    cv::cuda::PtrStepSzInt2 _d_ring_output;

#ifdef DEBUG_WRITE_GAUSSIAN_AS_PGM
    float*                  _h_debug_smooth;
#endif // DEBUG_WRITE_GAUSSIAN_AS_PGM
#ifdef DEBUG_WRITE_MAP_AS_PGM
    unsigned char*          _h_debug_map;
#endif // DEBUG_WRITE_MAP_AS_PGM
    unsigned char*          _h_debug_hyst_edges;
public: // HACK FOR DEBUGGING
    // unsigned char*          _h_plane;
    // uint32_t*               _h_debug_mag;
    // unsigned char*          _h_debug_edges;
    cv::cuda::PtrStepSzb    _h_plane;
    cv::cuda::PtrStepSz16s  _h_dx;
    cv::cuda::PtrStepSz16s  _h_dy;
    cv::cuda::PtrStepSz32u  _h_mag;
    cv::cuda::PtrStepSzb    _h_edges;
private:
    cv::cuda::PtrStepSzInt2 _h_ring_output;

    Voting _vote;

    FrameTexture*  _texture;
    cudaEvent_t*   _wait_for_upload;

public:
    // if we run out of streams (there are 32), we may have to share
    // bool         _stream_inherited;
    cudaStream_t _stream;
    cudaStream_t _download_stream;

    cudaEvent_t  _stream_done;
    cudaEvent_t  _download_stream_done;

    struct {
        cudaEvent_t  plane;
        cudaEvent_t  dxdy;
        cudaEvent_t  magmap;
        cudaEvent_t  edgecoords1;
        cudaEvent_t  edgecoords2;
        cudaEvent_t  descent1;
        cudaEvent_t  descent2;
    }            _download_ready_event;
};

}; // namespace popart

