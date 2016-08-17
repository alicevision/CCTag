/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "cuda/onoff.h"

#include <string>
#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include "cuda/tag_threads.h"
#include "cuda/tag_cut.h"
#include "cuda/geom_ellipse.h"
#include "cuda/geom_matrix.h"

#include "cctag/Params.hpp"
#include "cctag/Types.hpp"
#include "cctag/ImageCut.hpp"
#include "cctag/geometry/Ellipse.hpp"
#include "cctag/geometry/Point.hpp"

namespace cctag { namespace logtime { struct Mgmt; } };

namespace popart
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h
class NearbyPoint;

class TagPipe
{
    std::vector<Frame*>         _frame;
    const cctag::Parameters&    _params;
    TagThreads                  _threads;
    std::vector<cudaStream_t>   _tag_streams;

public:
    TagPipe( const cctag::Parameters& params );

    void initialize( const uint32_t pix_w,
                     const uint32_t pix_h,
                     cctag::logtime::Mgmt* durations );
    void release( );
    void load( unsigned char* pix );
    void tagframe( );
    void handleframe( int layer );

    void convertToHost( size_t                          layer,
                        cctag::EdgePointCollection&     edgeCollection,
                        std::vector<cctag::EdgePoint*>& seeds);

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
        const popart::geometry::ellipse&    outerEllipse, // in
        const float2&                       center,       // in
        const int                           vCutSize,     // in
        const cctag::Parameters&            params );     // in

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

    void makeCudaStreams( int numTags );

    void debug( unsigned char* pix,
                const cctag::Parameters& params );

    static void debug_cpu_origin( int                      layer,
                                  const cv::Mat&           img,
                                  const cctag::Parameters& params );

    static void debug_cpu_edge_out( int                      layer,
                                    const cv::Mat&           edges,
                                    const cctag::Parameters& params );

    static void debug_cpu_dxdy_out( TagPipe*                 pipe,
                                    int                      layer,
                                    const cv::Mat&           cpu_dx,
                                    const cv::Mat&           cpu_dy,
                                    const cctag::Parameters& params );

private:
    // implemented in frame_11_identify.cu
    /* to reuse various image-sized buffers, but retrieve their
     * bytesize to ensure that the new types fit into the
     * already allocated space.
     */
    identification::CutStruct*   _d_cut_struct;
    identification::CutStruct*   _h_cut_struct;
    NearbyPoint*                 _d_nearby_point;
    identification::CutSignals*  _d_cut_signals;
    int                          _num_cut_struct;
    int                          _num_nearby_point;
    int                          _num_cut_signals;

    intptr_t                     _d_cut_struct_end;
    intptr_t                     _h_cut_struct_end;
    intptr_t                     _d_nearby_point_end;
    intptr_t                     _d_cut_signals_end;

public:
    void checkTagAllocations( const int                numTags,
                              const cctag::Parameters& params );
private:
    void allocCutStructBuffer( int n );
    void allocNearbyPointBuffer( int n );
    void allocSignalBuffer( int n );
    void freeCutStructBuffer( );
    void freeNearbyPointBuffer( );
    void freeSignalBuffer( );

    size_t                       getCutStructBufferByteSize( ) const;
    identification::CutStruct*   getCutStructBufferDev( ) const;
    identification::CutStruct*   getCutStructBufferHost( ) const;
    size_t                       getNearbyPointGridBufferByteSize( ) const;
    NearbyPoint*                 getNearbyPointGridBuffer( int offset ) const;
    size_t                       getSignalBufferByteSize( ) const;
    identification::CutSignals*  getSignalBuffer( bool& success ) const;
    void                         clearSignalBuffer( );

    __host__
    bool imageCenterRetrieve(
        const int                           tagIndex,          // in
        cudaStream_t                        tagStream,         // in
        float2&                             bestPointOut,      // out
        float&                              bestResidual,      // out
        popart::geometry::matrix3x3&        bestHomographyOut, // out
        const cctag::Parameters&            params,            // in
        NearbyPoint*                        cctag_pointer_buffer );

    // implemented in frame_11_identify.cu
    __host__
    void idCostFunction(
        const int                           tagIndex,
        const int                           debug_numTags,
        cudaStream_t                        tagStream,
        int                                 iterations,
        const popart::geometry::ellipse&    ellipse,
        const float2                        center,
        const int                           vCutSize,     // in
        float                               currentNeighbourSize,
        const cctag::Parameters&            params,
        NearbyPoint*                        cctag_pointer_buffer );

};

}; // namespace popart

