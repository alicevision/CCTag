#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "onoff.h"

#include "cctag/Params.hpp"
#include "cctag/Types.hpp"
#include "cctag/ImageCut.hpp"
#include "cuda/geom_ellipse.h"
#include "cuda/framemeta.h"
#include "cuda/ptrstep.h"
#include "cuda/cuda_edge_point.h"
#include "cuda/edge_list.h"

#define RESERVE_MEM_MAX_CROWNS  5

/* This must replace params._maxEdges. That configuration variable
 * is definitely not big enough for finding all edge points in a 1K
 * image.
 */
#define EDGE_POINT_MAX                  1000000

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

namespace popart {
namespace identification {
// locally defined in frame_ident.cu only
struct CutStruct;
struct CutSignals;
} // identification
struct NearbyPoint;

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
    Frame( uint32_t width, uint32_t height, int my_layer, cudaStream_t download_stream, int my_pipe = 0 );
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

    // called by every thread, unpins uploaded image in frame 0
    void uploadComplete( );

    // Create a texture object this frame.
    // The caller must ensure that the Kind of texture object makes sense.
    void createTexture( FrameTexture::Kind kind );

    void deleteTexture( ); // Delete it. Done anyway in the destructor.

    // initialize this frame from other's normalized texture
    // void fillFromFrame( Frame& src );

    inline cudaTextureObject_t getTex( ) {
        assert( _texture );
        return _texture->getTex( );
    }

    void        allocUploadEvent( );
    void        deleteUploadEvent( );
    void        addUploadEvent( );
    cudaEvent_t getUploadEvent( );

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

    // implemented in frame_01_tex.cu
    void fillFromTexture( Frame& src );

    // implemented in frame_01_tex.cu
    void applyPlaneDownload( );

    // implemented in frame_02_gaussian.cu
    void applyGauss( const cctag::Parameters& param );

    // implemented in frame_02_gaussian.cu
    void applyGaussDownload( );

    // implemented in frame_03_magmap.cu
    void applyMag( );

    // implemented in frame_03_magmap.cu
    void applyMagDownload( );

    // implemented in frame_04_hyst.cu
    void applyHyst( );

    // implemented in frame_05_thin.cu
    void applyThinning( );
    void applyThinDownload( );

    // implemented in frame_06_graddesc.cu
    bool applyDesc( );

    // implemented in frame_07a_vote_line.cu
    bool applyVoteConstructLine( );

    // implemented in frame_07b_vote_sort_uniq_dp.cu
    // implemented in frame_07b_vote_sort_uniq_nodp.cu
    bool applyVoteSortUniq( );

#ifndef USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
private:
    // implemented in frame_07b_vote_sort_nodp.cu
    // called by applyVoteSortUniqNoDP
    bool applyVoteSortNoDP( );
    void applyVoteUniqNoDP( );
public:
#endif // not USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ

    // implemented in frame_07c_eval.cu
    bool applyVoteEval( );

    // implemented in frame_07d_vote_if.cu
    bool applyVoteIf( );

    // implemented in frame_07e_graddesc.cu
    void applyVoteDownload( );

    // implemented in frame_07_vote.cu
    void applyVote( );

    // implemented in frame_export.cu
    bool applyExport( cctag::EdgePointCollection& out_edges,
                      std::vector<cctag::EdgePoint*>& out_seedlist);

    cv::Mat* getPlane( ) const;
    cv::Mat* getDx( ) const;
    cv::Mat* getDy( ) const;
    cv::Mat* getMag( ) const;
    cv::Mat* getEdges( ) const;

protected:
    // implemented in frame_11_identify.cu
    /* to reuse various image-sized buffers, but retrieve their
     * bytesize to ensure that the new types fit into the
     * already allocated space.
     */
    size_t                               getCutStructBufferByteSize( ) const;
    popart::identification::CutStruct*   getCutStructBuffer( ) const;
    popart::identification::CutStruct*   getCutStructBufferHost( ) const;
    size_t                               getNearbyPointBufferByteSize( ) const;
    popart::NearbyPoint*                 getNearbyPointBuffer( ) const;
    size_t                               getSignalBufferByteSize( ) const;
    popart::identification::CutSignals*  getSignalBuffer( ) const;
    void                                 clearSignalBuffer( );

    friend class TagPipe;

public:
    // implemented in frame_11_identify.cu
    __host__
    void imageCenterOptLoop(
        const int                           tagIndex,     // in
        cudaStream_t                        tagStream,    // in
        const popart::geometry::ellipse&    outerEllipse, // in
        const float2&                       center,       // in
        const int                           vCutSize,     // in
        const cctag::Parameters&            params,       // in
        NearbyPoint*                        cctag_pointer_buffer );

    __host__
    bool imageCenterRetrieve(
        const int                           tagIndex,          // in
        cudaStream_t                        tagStream,         // in
        float2&                             bestPointOut,      // out
        popart::geometry::matrix3x3&        bestHomographyOut, // out
        const cctag::Parameters&            params,            // in
        NearbyPoint*                        cctag_pointer_buffer );

private:
    // implemented in frame_11_identify.cu
    __host__
    void idCostFunction(
        const int                           tagIndex,
        cudaStream_t                        tagStream,
        int                                 iterations,
        const popart::geometry::ellipse&    ellipse,
        const float2                        center,
        const int                           vCutSize,     // in
        float                               currentNeighbourSize,
        const cctag::Parameters&            params,
        NearbyPoint*                        cctag_pointer_buffer );

public:
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

    FrameMetaPtr            _meta; // lots of small variables

    cv::cuda::PtrStepSzb    _d_plane;
    cv::cuda::PtrStepSzf    _d_intermediate;
    cv::cuda::PtrStepSzf    _d_smooth;
    cv::cuda::PtrStepSz16s  _d_dx; // cv::cuda::PtrStepSzf _d_dx;
    cv::cuda::PtrStepSz16s  _d_dy; // cv::cuda::PtrStepSzf _d_dy;
    cv::cuda::PtrStepSz32u  _d_mag;
    cv::cuda::PtrStepSzb    _d_map;
    cv::cuda::PtrStepSzb    _d_hyst_edges;
    cv::cuda::PtrStepSzb    _d_edges;

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
    // Stores all edge points. Valid after thinning.
    EdgeList<CudaEdgePoint> _edgepoints;

    /* The following contain integer values that
     * represent an index into _edgepoints:
     *   _d_edgepoint_index_table - confirmed
     *   _voters - confirmed
     *   _v_chosen_idx
     *   _inner_points
     *   _interm_inner_points
     * There is no longer any indirection like
     * _inner_points->_voters->_edgepoints
     */
    cv::cuda::PtrStepSz32s  _d_edgepoint_map; // 2D pos -> index in _edgepoints

    // Stores all points that are recognized as potential voters
    // in gradient descent.
    EdgeList<int>          _voters; // voter index -> index in _edgepoints
    float*                 _v_chosen_flow_length;

    /** for every index in _voters, _voting_for contains not the voter's own index
     *  but the index of the CudaEdgePoint for which it voted, or -1 if it didn't vote
     */
    EdgeList<int>          _voting_for;

    EdgeList<int>          _inner_points;
    EdgeList<int>          _interm_inner_points;

    /* A single int allocated on the device as an intermediate
     * value. Some CUB calls needs such a thing. Waste of space
     * but necessary.
     */
    int*                   _d_interm_int;

    FrameTexture*        _texture;
    cudaEvent_t          _wait_for_upload;
    const unsigned char* _image_to_upload;

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

