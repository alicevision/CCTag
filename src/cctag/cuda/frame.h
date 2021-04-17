/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <assert.h>
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda_types.hpp>

#include "onoff.h"

#include "cctag/Params.hpp"
#include "cctag/Types.hpp"
#include "cctag/ImageCut.hpp"
#include "frame_07_vote.h"
#include "triple_point.h"
#include "cctag/cuda/geom_ellipse.h"
#include "cctag/cuda/framemeta.h"
#include "cctag/cuda/ptrstep.h"

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

namespace cctag {
// namespace identification {
// // locally defined in frame_ident.cu only
// struct CutStruct;
// struct CutSignals;
// } // identification
// struct NearbyPoint;

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
    explicit FrameTexture( const cv::cuda::PtrStepSzb& plane );
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
    Frame( uint32_t width, uint32_t height, int my_layer, cudaStream_t download_stream, int pipe_id );
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

    cv::cuda::PtrStepSzb& getPlaneDev( ) { return _d_plane; }

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

    // implemented in frame_07c_eval.cu
    bool applyVoteEval( );

    // implemented in frame_07d_vote_if.cu
    bool applyVoteIf( );

    // implemented in frame_07e_graddesc.cu
    void applyVoteDownload( );

    // implemented in frame_07_vote.cu
    void applyVote( );

    // implemented in frame_link.cu
    void applyLink( const cctag::Parameters& param );

    // implemented in frame_export.cu
    bool applyExport( cctag::EdgePointCollection& out_edges,
                      std::vector<cctag::EdgePoint*>& out_seedlist,
                      const int max_edge_pt );

    cv::Mat* getPlane( ) const;
    cv::Mat* getDx( ) const;
    cv::Mat* getDy( ) const;
    cv::Mat* getMag( ) const;
    cv::Mat* getEdges( ) const;

    friend class TagPipe;

public:
    static void writeInt2Array( const char* filename, const int2* array, uint32_t sz );
    static void writeTriplePointArray( const char* filename, const TriplePoint* array, uint32_t sz );

    void writeHostDebugPlane( std::string filename, const cctag::Parameters& params );

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
    // Stores coordinates of all edges. Valid after thinning.
    EdgeList<short2>        _all_edgecoords;

    // Stores all points that are recognized as potential voters
    // in gradient descent.
    EdgeList<TriplePoint>  _voters;
    float*                 _v_chosen_flow_length;
    EdgeList<int>          _v_chosen_idx;

    EdgeList<int>          _inner_points;
    EdgeList<int>          _interm_inner_points;

    Voting _vote;

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

}; // namespace cctag

