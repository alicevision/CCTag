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

#define NUM_ID_STREAMS 8

namespace cctag { namespace logtime { struct Mgmt; } };

namespace popart
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

private:
    struct ImageCenter
    {
        bool                            _valid;
        const int                       _tagIndex;     // in
        const int                       _debug_numTags; // in - only for debugging
        const popart::geometry::ellipse _outerEllipse; // in
        popart::geometry::matrix3x3     _mT;
        popart::geometry::matrix3x3     _mInvT;
        const float                     _maxSemiAxis; // in
        const float2                    _center;       // in
        const int                       _vCutSize;     // in
        int                             _iterations;
        float                           _transformedEllipseMaxRadius;
        NearbyPoint*                    _cctag_pointer_buffer; // out

        ImageCenter( const int                       tagIndex,
                     const int                       debug_numTags,
                     const popart::geometry::ellipse outerEllipse,
                     const float2&                   center,
                     const int                       vCutSize,
                     NearbyPoint*                    cctag_pointer_buffer,
                     const cctag::Parameters& params )
            : _valid( true )
            , _tagIndex( tagIndex )
            , _debug_numTags( debug_numTags )
            , _outerEllipse( outerEllipse )
            , _maxSemiAxis( std::max( outerEllipse.a(), outerEllipse.b() ) )
            , _center( center )
            , _vCutSize( vCutSize )
            , _iterations( 0 )
            , _cctag_pointer_buffer( cctag_pointer_buffer )
        {
            const size_t gridNSample   = params._imagedCenterNGridSample;
            float        neighbourSize = params._imagedCenterNeighbourSize;

            if( _vCutSize < 2 ) {
                _valid = false;
                return;
            }

            if( v._vCutSize != 22 ) {
                cerr << __FILE__ << ":" << __LINE__ << endl
                     << "    " << __func__ << " is called from CPU code with vCutSize " << v._vCutSize << " instead of 22" << endl;
                if( v._vCutSize > 22 ) {
                    exit( -1 );
                }
            }

            /* Determine the number of iterations by iteration */
            while( neighbourSize * _maxSemiAxis > 0.02 ) {
                _iterations += 1;
                neighbourSize /= (float)((gridNSample-1)/2) ;
            }

            _outerEllipse.makeConditionerFromEllipse( _mT );

            bool good = _mT.invert( _mInvT );
            if( not good ) {
                std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                          << "    Conditioner matrix extracted from ellipse is not invertable" << std::endl
                          << "    Program logic error. Requires analysis before fixing." << std::endl
                          << std::endl;
                _valid = false;

                return;
            }

            popart::geometry::ellipse transformedEllipse;
            _outerEllipse.projectiveTransform( _mInvT, transformedEllipse );
            _transformedEllipseMaxRadius = std::max( transformedEllipse.a(), transformedEllipse.b() );
        }

        void setInvalid( )
        {
            _valid = false;
        }
    };

    ImageCenter* _d_image_center_opt_input;
    ImageCenter* _h_image_center_opt_input;

public:
    __host__
    void imageCenterOptPrepare(
        const int                                  tagIndex,
        const int                                  debug_numTags, // in - only for debugging
        const cctag::numerical::geometry::Ellipse& ellipse,
        const cctag::Point2d<Eigen::Vector3f>&     center,
        const int                                  vCutSize,
        const cctag::Parameters&                   params,
        NearbyPoint*                               cctag_pointer_buffer );

    __host__
    void imageCenterOpt( );

private:
    __host__
    void imageCenterOptLoop( );


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
        popart::geometry::matrix3x3&        bestHomographyOut, // out
        const cctag::Parameters&            params,            // in
        NearbyPoint*                        cctag_pointer_buffer );

    // implemented in frame_11_identify.cu
    __host__
    void idCostFunction( std::vector<bool>& success );

};

}; // namespace popart

