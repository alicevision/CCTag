#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "onoff.h"

#include "cctag/params.hpp"
#include "cctag/types.hpp"
#include "cctag/ImageCut.hpp"
#include "frame_vote.h"
#include "triple_point.h"
#include "cuda/geom_ellipse.h"

#define RESERVE_MEM_MAX_CROWNS  5

#define EDGE_LINKING_HOST_SIDE

#define EDGE_LINKING_MAX_EDGE_LENGTH        100
#define EDGE_LINKING_MAX_ARCS             10000
#define EDGE_LINKING_MAX_RING_BUFFER_SIZE    40

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
namespace identification {
// locally defined in frame_ident.cu only
struct CutStruct;
struct NearbyPoint;
} // identification

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
 * FrameTexture
 * Every from has one of these structures. It is allocated in
 * pinned host memory, but is also mapped to the device.
 */
struct FrameMeta
{
    int   hysteresis_block_counter;
    int   connect_component_block_counter;
    int   ring_counter;
    int   ring_counter_max;
    float identification_result;
    int   identification_resct;

    static void alloc( FrameMeta** host, FrameMeta** device );
    static void release( FrameMeta* host );
};

/*************************************************************
 * Frame
 * The basic structure for managing data stored on the GPU
 *************************************************************/
class Frame
{
public:
    // create continuous device memory, enough for @layers copies of @width x @height
    Frame( uint32_t width, uint32_t height, int my_layer, cudaStream_t download_stream );
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
    void applyPlaneDownload( const cctag::Parameters& param );

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
    bool applyDesc0( const cctag::Parameters& param );
#ifdef USE_SEPARABLE_COMPILATION_IN_GRADDESC
    bool applyDesc1( const cctag::Parameters& param );
    bool applyDesc2( const cctag::Parameters& param );
    bool applyDesc3( const cctag::Parameters& param );
    bool applyDesc4( const cctag::Parameters& param );
    bool applyDesc5( const cctag::Parameters& param );
    bool applyDesc6( const cctag::Parameters& param );
#else // USE_SEPARABLE_COMPILATION
    bool applyDesc( const cctag::Parameters& param );
#endif // USE_SEPARABLE_COMPILATION

    // implemented in frame_graddesc.cu
    void applyDescDownload( const cctag::Parameters& param );

    // implemented in frame_vote.cu
    void applyVote( const cctag::Parameters& param );

    // implemented in frame_vote_sort_nodp.cu
    // called by applyVote
    bool applyVoteSortNoDP( const cctag::Parameters& params );

    // implemented in frame_vote_uniq_nodp.cu
    // called by applyVote
    void applyVoteUniqNoDP( const cctag::Parameters& params );

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

private:
    // implemented in frame_ident.cu
    /* to reuse various image-sized buffers, but retrieve their
     * bytesize to ensure that the new types fit into the
     * already allocated space.
     */
    size_t                               getCutStructBufferByteSize( ) const;
    popart::identification::CutStruct*   getCutStructBuffer( ) const;
    popart::identification::CutStruct*   getCutStructBufferHost( ) const;
    size_t                               getNearbyPointBufferByteSize( ) const;
    popart::identification::NearbyPoint* getNearbyPointBuffer( ) const;
    size_t                               getSignalBufferByteSize( ) const;
    float*                               getSignalBuffer( ) const;
    void                                 clearSignalBuffer( );

    // implemented in frame_ident.cu
    void uploadCuts( std::vector<cctag::ImageCut>& vCuts );

public:
    // implemented in frame_ident.cu
    __host__
    double idCostFunction( const popart::geometry::ellipse& ellipse,
                           const float2                     center,
                           const int                        vCutsSize,
                           const int                        vCutMaxVecLen,
                           bool&                            readable );

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

    FrameMeta*              _h_meta; // pointer to pinned mem
    FrameMeta*              _d_meta; // mapping to device mem

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

#ifdef DEBUG_WRITE_MAP_AS_PGM
    unsigned char*          _h_debug_map;
#endif // DEBUG_WRITE_MAP_AS_PGM
    unsigned char*          _h_debug_hyst_edges;
public: // HACK FOR DEBUGGING
    cv::cuda::PtrStepSzb    _h_plane;
    cv::cuda::PtrStepSz16s  _h_dx;
    cv::cuda::PtrStepSz16s  _h_dy;
    cv::cuda::PtrStepSz32u  _h_mag;
    cv::cuda::PtrStepSzb    _h_edges;

    cv::cuda::PtrStepSzf    _h_intermediate; // copies layout of _d_intermediate
private:
    cv::cuda::PtrStepSzInt2 _h_ring_output;

    Voting _vote;

    FrameTexture*  _texture;
    cudaEvent_t*   _wait_for_upload;

public:
    // if we run out of streams (there are 32), we may have to share
    // bool         _stream_inherited;
    cudaStream_t _stream;
    bool         _private_download_stream;
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

