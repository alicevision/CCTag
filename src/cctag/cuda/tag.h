/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "cctag/cuda/onoff.h"

#include <string>
#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/core.hpp>
#include <cctag/cuda/cctag_cuda_runtime.h>

#include "cctag/cuda/tag_threads.h"
#include "cctag/cuda/tag_cut.h"
#include "cctag/cuda/geom_ellipse.h"
#include "cctag/cuda/geom_matrix.h"

#include "cctag/Params.hpp"
#include "cctag/Types.hpp"
#include "cctag/ImageCut.hpp"
#include "cctag/geometry/Ellipse.hpp"
#include "cctag/geometry/Point.hpp"

#define NUM_ID_STREAMS 8

namespace cctag { namespace logtime { struct Mgmt; } };

namespace cctag
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h
class NearbyPoint;
class NearbyPointGrid;

class TagPipe
{
    static int                  _tag_id_running_number;

    int                         _tag_id;
    std::vector<Frame*>         _frame;
    const cctag::Parameters&    _params;
    TagThreads                  _threads;
    cudaStream_t                _tag_streams[NUM_ID_STREAMS];
    cudaEvent_t                 _uploaded_event;

public:
    TagPipe( const cctag::Parameters& params );

    inline int getId() const { return _tag_id; }

    void initialize( const uint32_t pix_w,
                     const uint32_t pix_h,
                     cctag::logtime::Mgmt* durations );
    void release( );
    void load( int frameId, unsigned char* pix );
    void tagframe( );
    void handleframe( int layer );

    void convertToHost( size_t                          layer,
                        cctag::EdgePointCollection&     edgeCollection,
                        std::vector<cctag::EdgePoint*>& seeds,
                        const int                       max_edge_ct );

    inline std::size_t getNumOctaves( ) const {
        return _frame.size();
    }

    uint32_t getWidth(  size_t layer ) const;
    uint32_t getHeight( size_t layer ) const;

    cv::Mat* getPlane( size_t layer ) const;
    cv::Mat* getDx( size_t layer ) const;
    cv::Mat* getDy( size_t layer ) const;
    cv::Mat* getMag( size_t layer ) const;
    cv::Mat* getEdges( size_t layer ) const;

public:
    __host__
    void imageCenterOptLoop(
        const int                                  tagIndex,
        const int                                  debug_numTags, // in - only for debugging
        const cctag::numerical::geometry::Ellipse& ellipse,
        const cctag::Point2d<Eigen::Vector3f>&     center,
        const int                                  vCutSize,
        const cctag::Parameters&                   params,
        NearbyPoint*                               cctag_pointer_buffer );

private:
    __host__
    void imageCenterOptLoop(
        const int                           tagIndex,     // in
        const int                           debug_numTags, // in - only for debugging
        cudaStream_t                        tagStream,    // in
        const cctag::geometry::ellipse&    outerEllipse, // in
        const float2&                       center,       // in
        const int                           vCutSize,     // in
        const cctag::Parameters&            params,       // in
        NearbyPoint*                        cctag_pointer_buffer ); // out

public:
    bool imageCenterRetrieve(
        const int                        tagIndex,
        cctag::Point2d<Eigen::Vector3f>& center,
        float&                           bestResidual,
        Eigen::Matrix3f&                 bestHomographyOut,
        const cctag::Parameters&         params,
        NearbyPoint*                     cctag_pointer_buffer );

    // size_t getSignalBufferByteSize( int level ) const;

    void uploadCuts( int                                 numTags,
                     const std::vector<cctag::ImageCut>* vCuts,
                     const cctag::Parameters&            params );

private:
    // implemented in frame_11_identify.cu
    /* to reuse various image-sized buffers, but retrieve their
     * bytesize to ensure that the new types fit into the
     * already allocated space.
     */
    CutStructGrid*   _d_cut_struct_grid;
    CutStructGrid*   _h_cut_struct_grid;
    NearbyPointGrid* _d_nearby_point_grid;
    CutSignalGrid*   _d_cut_signal_grid;
    int              _num_cut_struct_grid;
    int              _num_nearby_point_grid;
    int              _num_cut_signal_grid;

public:
    void checkTagAllocations( const int                numTags,
                              const cctag::Parameters& params );
private:
    void reallocNearbyPointGridBuffer( int numTags );
    void freeNearbyPointGridBuffer( );
    NearbyPointGrid* getNearbyPointGridBuffer( int tagIndex ) const;

    void reallocCutStructGridBuffer( int numTags );
    void freeCutStructGridBuffer( );
    CutStructGrid* getCutStructGridBufferDev( int tagIndex ) const;
    CutStructGrid* getCutStructGridBufferHost( int tagIndex ) const;

    void reallocSignalGridBuffer( int numTags );
    void freeSignalGridBuffer( );
    CutSignalGrid* getSignalGridBuffer( int tagIndex ) const;

    __host__
    bool imageCenterRetrieve(
        const int                           tagIndex,          // in
        cudaStream_t                        tagStream,         // in
        float2&                             bestPointOut,      // out
        float&                              bestResidual,      // out
        cctag::geometry::matrix3x3&        bestHomographyOut, // out
        const cctag::Parameters&            params,            // in
        NearbyPoint*                        cctag_pointer_buffer );

    // implemented in frame_11_identify.cu
    __host__
    bool idCostFunction(
        const int                           tagIndex,
        const int                           debug_numTags,
        cudaStream_t                        tagStream,
        int                                 iterations,
        const cctag::geometry::ellipse&    ellipse,
        const float2                        center,
        const int                           vCutSize,     // in
        float                               currentNeighbourSize,
        const cctag::Parameters&            params );

};

}; // namespace cctag

