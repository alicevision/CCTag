/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "tag.h"
#include "frame.h"
#include "frameparam.h"
#include "debug_macros.hpp"
#include "keep_time.hpp"
#include "pinned_counters.h"
#include <sstream>
#include <iostream>
#include <fstream>

#include "debug_image.h"
#include "cctag/utils/Talk.hpp"
#include "geom_ellipse.h"

#include "onoff.h"
#include "tag_threads.h"
#include "tag_cut.h"

#if 1
    __device__ __host__
    inline int validate( const char* file, int line, int input, int reference )
    {
#if 1
        return min( input, reference );
#else
        if( input < reference ) {
            printf( "%s:%d Divergence: run-time value %d < conf %d\n", file, line, input, reference );
            return input;
        }
        if( input > reference ) {
            printf( "%s:%d Divergence: run-time value %d > conf %d\n", file, line, input, reference );
            return reference;
        }
        return reference;
#endif
    }
    #define STRICT_CUTSIZE(sz) validate( __FILE__, __LINE__, sz, 22 )
    #define STRICT_SAMPLE(sz)  validate( __FILE__, __LINE__, sz, 5 )
    #define STRICT_SIGSIZE(sz) validate( __FILE__, __LINE__, sz, 100 )
#else
    #define STRICT_CUTSIZE(sz) sz
    #define STRICT_SAMPLE(sz)  sz
    #define STRICT_SIGSIZE(sz) sz
#endif

using namespace std;

namespace cctag
{
int TagPipe::_tag_id_running_number = 0;

__host__
TagPipe::TagPipe( const cctag::Parameters& params )
    : _params( params )
    , _d_cut_struct_grid( 0 )
    , _h_cut_struct_grid( 0 )
    , _d_nearby_point_grid( 0 )
    , _d_cut_signal_grid( 0 )
    , _num_cut_struct_grid( 0 )
    , _num_nearby_point_grid( 0 )
    , _num_cut_signal_grid( 0 )
{
    _tag_id = _tag_id_running_number;
    _tag_id_running_number++;
    cerr << "Creating TagPipe " << _tag_id << endl;
}

__host__
void TagPipe::initialize( const uint32_t pix_w,
                          const uint32_t pix_h,
                          cctag::logtime::Mgmt* durations )
{
    cerr << "Initializing TagPipe " << _tag_id << endl;
    PinnedCounters::init( getId() );

    static bool tables_initialized = false;
    if( ! tables_initialized ) {
        tables_initialized = true;
        Frame::initGaussTable( );
        Frame::initThinningTable( );
    }

    FrameParam::init( _params );

    int num_layers = _params._numberOfMultiresLayers;
    _frame.reserve( num_layers );

    uint32_t w = pix_w;
    uint32_t h = pix_h;
    cctag::Frame* f;
#ifdef USE_ONE_DOWNLOAD_STREAM
    cudaStream_t download_stream = 0;
    for( int i=0; i<num_layers; i++ ) {
        _frame.push_back( f = new cctag::Frame( w, h, i, download_stream, getId() ) ); // sync
        if( i==0 ) { download_stream = f->_download_stream; assert( download_stream != 0 ); }
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
#else
    for( int i=0; i<num_layers; i++ ) {
        _frame.push_back( f = new cctag::Frame( w, h, i, 0, getId() ) ); // sync
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
#endif

    _frame[0]->createTexture( cctag::FrameTexture::normalized_uchar_to_float); // sync
    _frame[0]->allocUploadEvent( ); // sync

    for( int i=0; i<num_layers; i++ ) {
        _frame[i]->allocRequiredMem( _params ); // sync
    }

    for( int i=0; i<NUM_ID_STREAMS; i++ ) {
        POP_CUDA_STREAM_CREATE( &_tag_streams[i] );
    }
    cudaEventCreate( &_uploaded_event );

    _threads.init( this, num_layers );
}

__host__
void TagPipe::release( )
{
    cerr << "Releasing TagPipe " << _tag_id << endl;
    cudaEventDestroy( _uploaded_event );
    for( int i=0; i<NUM_ID_STREAMS; i++ ) {
        POP_CUDA_STREAM_DESTROY( _tag_streams[i] );
    }

    PinnedCounters::release( getId() );
}

__host__
uint32_t TagPipe::getWidth(  size_t layer ) const
{
    return _frame[layer]->getWidth();
}

__host__
uint32_t TagPipe::getHeight( size_t layer ) const
{
    return _frame[layer]->getHeight();
}

__host__
void TagPipe::load( int frameId, unsigned char* pix )
{
    cerr << "Loading image " << frameId << " into TagPipe " << _tag_id << endl;
    _frame[0]->upload( pix ); // async
    _frame[0]->addUploadEvent( ); // async
}

__host__
void TagPipe::tagframe( )
{
    _threads.oneRound( );
}

__host__
void TagPipe::handleframe( int i )
{
    _frame[i]->initRequiredMem( ); // async

    cudaEvent_t ev = _frame[0]->getUploadEvent( ); // async

    if( i > 0 ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->uploadComplete( ); // unpin image
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
    }

    // note: without visual debug, only level 0 performs download
    _frame[i]->applyPlaneDownload(); // async

    _frame[i]->applyGauss( _params ); // async
    _frame[i]->applyMag();  // async

    _frame[i]->applyHyst();  // async
    _frame[i]->applyThinning();  // async

    _frame[i]->applyDesc();  // async
    _frame[i]->applyVoteConstructLine();  // async
    _frame[i]->applyVoteSortUniq();  // async

    _frame[i]->applyGaussDownload(); // async
    _frame[i]->applyMagDownload();
    _frame[i]->applyThinDownload(); // sync

    _frame[i]->applyVoteEval();  // async
    _frame[i]->applyVoteIf();  // async

    _frame[i]->applyVoteDownload();   // sync!

    cudaStreamSynchronize( _frame[i]->_stream );
    cudaStreamSynchronize( _frame[i]->_download_stream );
}

__host__
void TagPipe::convertToHost( size_t                          layer,
                             cctag::EdgePointCollection&     edgeCollection,
                             std::vector<cctag::EdgePoint*>& seeds,
                             const int                       max_edge_ct )
{
    assert( layer < _frame.size() );

    _frame[layer]->applyExport( edgeCollection, seeds, max_edge_ct );

}

__host__
cv::Mat* TagPipe::getPlane( size_t layer ) const
{
    assert( layer < getNumOctaves() );
    return _frame[layer]->getPlane();
}

__host__
cv::Mat* TagPipe::getDx( size_t layer ) const
{
    assert( layer < getNumOctaves() );
    return _frame[layer]->getDx();
}

__host__
cv::Mat* TagPipe::getDy( size_t layer ) const
{
    assert( layer < getNumOctaves() );
    return _frame[layer]->getDy();
}

__host__
cv::Mat* TagPipe::getMag( size_t layer ) const
{
    assert( layer < getNumOctaves() );
    return _frame[layer]->getMag();
}

__host__
cv::Mat* TagPipe::getEdges( size_t layer ) const
{
    assert( layer < getNumOctaves() );
    return _frame[layer]->getEdges();
}

__host__
void TagPipe::imageCenterOptLoop(
    const int                                  tagIndex,
    const int                                  debug_numTags,
    const cctag::numerical::geometry::Ellipse& ellipse,
    const cctag::Point2d<Eigen::Vector3f>&     center,
    const int                                  vCutSize,
    const cctag::Parameters&                   params,
    NearbyPoint*                               cctag_pointer_buffer )
{
    // cerr << __FILE__ << ":" << __LINE__ << " enter imageCenterOptLoop for tag " << tagIndex << " number of cuts is " << vCutSize << endl;
    cctag::geometry::ellipse e( ellipse.matrix()(0,0),
                                 ellipse.matrix()(0,1),
                                 ellipse.matrix()(0,2),
                                 ellipse.matrix()(1,0),
                                 ellipse.matrix()(1,1),
                                 ellipse.matrix()(1,2),
                                 ellipse.matrix()(2,0),
                                 ellipse.matrix()(2,1),
                                 ellipse.matrix()(2,2),
                                 ellipse.center().x(),
                                 ellipse.center().y(),
                                 ellipse.a(),
                                 ellipse.b(),
                                 ellipse.angle() );
    float2 f = make_float2( center.x(), center.y() );

    imageCenterOptLoop( tagIndex,
                        debug_numTags,
                        _tag_streams[tagIndex%NUM_ID_STREAMS],
                        e,
                        f,
                        vCutSize,
                        params,
                        cctag_pointer_buffer );
}

__host__
bool TagPipe::imageCenterRetrieve(
    const int                                  tagIndex,
    cctag::Point2d<Eigen::Vector3f>&           center,
    float&                                     bestResidual,
    Eigen::Matrix3f&                           bestHomographyOut,
    const cctag::Parameters&                   params,
    NearbyPoint*                               cctag_pointer_buffer )
{
    float2                      bestPoint;
    cctag::geometry::matrix3x3 bestHomography;

    bool success = imageCenterRetrieve( tagIndex,
                                        _tag_streams[tagIndex%NUM_ID_STREAMS],
                                        bestPoint,
                                        bestResidual,
                                        bestHomography,
                                        params,
                                        cctag_pointer_buffer );

    if( success ) {
        center.x() = bestPoint.x;
        center.y() = bestPoint.y;

    #pragma unroll
    for( int i=0; i<3; i++ ) {
        #pragma unroll
        for( int j=0; j<3; j++ ) {
                bestHomographyOut(i,j) = bestHomography(i,j);
            }
        }
    }
    return success;
}

/** How much GPU-side memory do we need?
 *  For each tag (numTags), we need gridNSample^2  nearby points.
 *  For each tag (numTags) and each nearby point (gridNSample^2), we
 *    need a signal buffer (sizeof(CutSignals)).
 *    Note that sizeof(CutSignals) must also be >=
 *    sizeof(float) * sampleCutLength + sizeof(uint32_t)
 *  For each tag (numTags), we need space for max cut size
 *    (params._numCutsInIdentStep) cuts without the signals.
 */
void TagPipe::checkTagAllocations( const int                numTags,
                                   const cctag::Parameters& params )
{
    const size_t gridNSample = STRICT_SAMPLE( params._imagedCenterNGridSample ); // 5
    const size_t numCuts     = STRICT_CUTSIZE( params._numCutsInIdentStep ); // 22

    reallocNearbyPointGridBuffer( numTags ); // each numTags is gridNSample^2 points
    reallocSignalGridBuffer( numTags );      // each numTags is gridNSample^2 * numCuts signals
    reallocCutStructGridBuffer( numTags );   // each numTags is numCuts cut structs
}

__host__
void TagPipe::uploadCuts( int                                 numTags,
                          const std::vector<cctag::ImageCut>* vCuts,
                          const cctag::Parameters&            params )
{
    if( numTags <= 0 || vCuts == 0 || vCuts->size() == 0 ) return;

    const int max_cuts_per_Tag = STRICT_CUTSIZE( params._numCutsInIdentStep );

    cerr << endl << "==== Uploading " << numTags << " tags ====" << endl;

    for( int tagIndex=0; tagIndex<numTags; tagIndex++ ) {
        // cerr << "    Tag " << tagIndex << " has " << vCuts[tagIndex].size() << " cuts" << endl;

        CutStructGrid* cutGrid = this->getCutStructGridBufferHost( tagIndex );

        if( STRICT_CUTSIZE( vCuts[tagIndex].size() ) > max_cuts_per_Tag ) {
            cerr << __FILE__ << "," << __LINE__ << ":" << endl
                 << "    Programming error: assumption that number of cuts for a single tag is < params._numCutsInIdentStep is wrong" << endl;
            exit( -1 );
        }

        std::vector<cctag::ImageCut>::const_iterator vit  = vCuts[tagIndex].begin();
        std::vector<cctag::ImageCut>::const_iterator vend = vCuts[tagIndex].end();

        for( int cut=0 ; vit!=vend; vit++, cut++ ) {
            cutGrid->getGrid(cut).start.x     = vit->start().x();
            cutGrid->getGrid(cut).start.y     = vit->start().y();
            cutGrid->getGrid(cut).stop.x      = vit->stop().x();
            cutGrid->getGrid(cut).stop.y      = vit->stop().y();
            cutGrid->getGrid(cut).beginSig    = vit->beginSig();
            cutGrid->getGrid(cut).endSig      = vit->endSig();
            cutGrid->getGrid(cut).sigSize     = STRICT_SIGSIZE( vit->imgSignal().size() );
        }
    }

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( this->getCutStructGridBufferDev( 0 ),
                                     this->getCutStructGridBufferHost( 0 ),
                                     numTags * sizeof(CutStructGrid),
                                     _tag_streams[0] );
    cudaEventRecord( _uploaded_event, _tag_streams[0] );
    for( int i=1; i<NUM_ID_STREAMS; i++ ) {
        cudaStreamWaitEvent( _tag_streams[i], _uploaded_event, 0 );
    }
}

}; // namespace cctag

