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
#include "cuda/geom_ellipse.h"

#include "cuda/onoff.h"
#include "cuda/tag_threads.h"
#include "cuda/tag_cut.h"

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

namespace popart
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
    _tag_id = _tag_id_running_number++;
    cerr << "Creating TagPipe " << _tag_id << endl;
}

__host__
void TagPipe::initialize( const uint32_t pix_w,
                          const uint32_t pix_h,
                          cctag::logtime::Mgmt* durations )
{
    cerr << "Initializing TagPipe " << _tag_id << endl;
    PinnedCounters::init( );

    static bool tables_initialized = false;
    if( not tables_initialized ) {
        tables_initialized = true;
        Frame::initGaussTable( );
        Frame::initThinningTable( );
    }

    FrameParam::init( _params );

    int num_layers = _params._numberOfMultiresLayers;
    _frame.reserve( num_layers );

    uint32_t w = pix_w;
    uint32_t h = pix_h;
    popart::Frame* f;
#ifdef USE_ONE_DOWNLOAD_STREAM
    cudaStream_t download_stream = 0;
    for( int i=0; i<num_layers; i++ ) {
        _frame.push_back( f = new popart::Frame( w, h, i, download_stream ) ); // sync
        if( i==0 ) { download_stream = f->_download_stream; assert( download_stream != 0 ); }
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
#else
    for( int i=0; i<num_layers; i++ ) {
        _frame.push_back( f = new popart::Frame( w, h, i, 0 ) ); // sync
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
#endif

    _frame[0]->createTexture( popart::FrameTexture::normalized_uchar_to_float); // sync
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

    PinnedCounters::release( );
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
void TagPipe::load( unsigned char* pix )
{
    cerr << "Loading image into TagPipe " << _tag_id << endl;
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
                             std::vector<cctag::EdgePoint*>& seeds)
{
    assert( layer < _frame.size() );

    _frame[layer]->applyExport( edgeCollection, seeds );

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
void TagPipe::debug( unsigned char* pix, const cctag::Parameters& params )
{
    DO_TALK( cerr << "Enter " << __FUNCTION__ << endl; )

    if( true ) {
        if( params._debugDir == "" ) {
            DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__
                << ": debugDir not set, not writing debug output" << endl; )
            return;
        } else {
            DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__ << ": debugDir is ["
                 << params._debugDir << "] using that directory" << endl; )
        }

        // This is a debug block

        int num_layers = _frame.size();

        for( int i=0; i<num_layers; i++ ) {
            _frame[i]->hostDebugDownload( params );
        }
        POP_SYNC_CHK;

        _frame[0]->hostDebugCompare( pix );

        for( int i=0; i<num_layers; i++ ) {
            std::ostringstream ostr;
            ostr << "gpu-" << i;
            _frame[i]->writeHostDebugPlane( ostr.str(), params );
        }
        POP_SYNC_CHK;
    }

    DO_TALK( cerr << "terminating in tagframe" << endl; )
    DO_TALK( cerr << "Leave " << __FUNCTION__ << endl; )
    // exit( 0 );
}

void TagPipe::debug_cpu_origin( int                      layer,
                                const cv::Mat&           img,
                                const cctag::Parameters& params )
{
    if( params._debugDir == "" ) {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__
            << ": debugDir not set, not writing debug output" << endl; )
        return;
    } else {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__ << ": debugDir is ["
            << params._debugDir << "] using that directory" << endl; )
    }

    ostringstream ascname;
    ascname << params._debugDir << "cpu-" << layer << "-img-ascii.txt";
    ofstream asc( ascname.str().c_str() );

    int cols = img.size().width;
    int rows = img.size().height;
    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            uint8_t pix = img.at<uint8_t>(y,x);
            asc << setw(3) << (int)pix << " ";
        }
        asc << endl;
    }
}

void TagPipe::debug_cpu_edge_out( int                      layer,
                                  const cv::Mat&           edges,
                                  const cctag::Parameters& params )
{
    if( params._debugDir == "" ) {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__
            << ": debugDir not set, not writing debug output" << endl; )
        return;
    } else {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__ << ": debugDir is ["
            << params._debugDir << "] using that directory" << endl; )
    }

    ostringstream filename;
    filename << params._debugDir
             << "cpu-" << layer << "-edges.ppm";

    cv::cuda::PtrStepSzb plane;
    plane.step = edges.size().width;
    plane.cols = edges.size().width;
    plane.rows = edges.size().height;
    if( plane.cols == 0 || plane.rows == 0 ) return;
    plane.data = new uint8_t[ plane.cols * plane.rows ];

    for( int y=0; y<plane.rows; y++ )
        for( int x=0; x<plane.cols; x++ ) {
            plane.ptr(y)[x] = edges.at<uint8_t>(y,x);
        }

    DebugImage::writePGM( filename.str(), plane );

    delete [] plane.data;
}

static void local_debug_cpu_dxdy_out( const char*                  dxdy,
                                      size_t                       level,
                                      const cv::Mat&               cpu,
                                      const cv::cuda::PtrStepSz16s gpu,
                                      const cctag::Parameters&     params )
{
    if( params._debugDir == "" ) {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__
            << ": debugDir not set, not writing debug output" << endl; )
        return;
    } else {
        DO_TALK( cerr << __FUNCTION__ << ":" << __LINE__ << ": debugDir is ["
            << params._debugDir << "] using that directory" << endl; )
    }

    if( cpu.size().width  != gpu.cols ) {
        cerr << __FILE__ << ":" << __LINE__
             << " Error: array width CPU " << cpu.size().width << " vs GPU " << gpu.cols << endl;
    }
    if( cpu.size().height != gpu.rows ) {
        cerr << __FILE__ << ":" << __LINE__
             << " Error: array height CPU " << cpu.size().height << " vs GPU " << gpu.rows << endl;
    }

    int cols = min( cpu.size().width, gpu.cols );
    int rows = min( cpu.size().height, gpu.rows );

    if( cols == 0 || rows == 0 ) return;

    cv::cuda::PtrStepSz16s plane;
    plane.step = cols * sizeof(int16_t);
    plane.cols = cols;
    plane.rows = rows;
    plane.data = new int16_t[ cols * rows ];

    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            int16_t cpu_val = cpu.at<int16_t>(y,x);
            int16_t gpu_val = gpu.ptr(y)[x];
            plane.ptr(y)[x] = (int16_t)gpu_val - (int16_t)cpu_val;
#if 0
            if( y < 4 || x < 4 || y >= rows-4 || x >= cols-4 ) {
                diffplane.ptr(y)[x] = 0;
            }
#endif
        }
    }
    ostringstream asc_f_diff;
    ostringstream img_f_diff;
    asc_f_diff << params._debugDir << "diffcpugpu-" << level << "-" << dxdy << "-ascii.txt";
    img_f_diff << params._debugDir << "diffcpugpu-" << level << "-" << dxdy << ".pgm";
    DebugImage::writePGMscaled( img_f_diff.str(), plane );
    DebugImage::writeASCII(     asc_f_diff.str(), plane );

    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            int16_t cpu_val   = cpu.at<int16_t>(y,x);
            plane.ptr(y)[x] = min<int16_t>( max<int16_t>( (int16_t)cpu_val, -255 ), 255 );
        }
    }

    ostringstream asc_f_cpu;
    ostringstream img_f_cpu;
    asc_f_cpu  << params._debugDir << "cpu-" << level << "-" << dxdy << "-ascii.txt";
    img_f_cpu  << params._debugDir << "cpu-" << level << "-" << dxdy << ".pgm";
    DebugImage::writePGMscaled( img_f_cpu.str(), plane );
    DebugImage::writeASCII(     asc_f_cpu.str(), plane );

    delete [] plane.data;
}

void TagPipe::debug_cpu_dxdy_out( TagPipe*                     pipe,
                                  int                          layer,
                                  const cv::Mat&               cpu_dx,
                                  const cv::Mat&               cpu_dy,
                                  const cctag::Parameters&     params )
{
    const cv::cuda::PtrStepSz16s gpu_dx = pipe->_frame[layer]->_h_dx;
    const cv::cuda::PtrStepSz16s gpu_dy = pipe->_frame[layer]->_h_dy;
    size_t                       level  = pipe->_frame[layer]->getLayer();

    local_debug_cpu_dxdy_out( "dx", level, cpu_dx, gpu_dx, params );
    local_debug_cpu_dxdy_out( "dy", level, cpu_dy, gpu_dy, params );
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
    popart::geometry::ellipse e( ellipse.matrix()(0,0),
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

    popart::geometry::matrix3x3 bestHomography;

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
    popart::geometry::matrix3x3 bestHomography;

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

}; // namespace popart

