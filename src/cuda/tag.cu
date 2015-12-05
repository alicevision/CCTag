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
#include "cctag/talk.hpp"
#include "cuda/geom_ellipse.h"
#include "cctag/algebra/matrix/Matrix.hpp"

#include "cctag/logtime.hpp"
#include "cuda/onoff.h"
#include "cuda/tag_threads.h"

using namespace std;

namespace popart
{
__host__
TagPipe::TagPipe( const cctag::Parameters& params )
    : _params( params )
{
}

__host__
void TagPipe::initialize( const uint32_t pix_w,
                          const uint32_t pix_h,
                          cctag::logtime::Mgmt* durations )
{
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

    _threads.init( this, num_layers );
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
#ifdef SHOW_DETAILED_TIMING
    KeepTime* time_gauss;
    KeepTime* time_mag;
    KeepTime* time_hyst;
    KeepTime* time_thin;
    KeepTime* time_desc;
    KeepTime* time_vote;
    time_gauss = new KeepTime( _frame[i]->_stream );
    time_mag   = new KeepTime( _frame[i]->_stream );
    time_hyst  = new KeepTime( _frame[i]->_stream );
    time_thin  = new KeepTime( _frame[i]->_stream );
    time_desc  = new KeepTime( _frame[i]->_stream );
    time_vote  = new KeepTime( _frame[i]->_stream );
#endif

    _frame[i]->initRequiredMem( ); // async

    cudaEvent_t ev = _frame[0]->getUploadEvent( ); // async

    if( i > 0 ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
    }

#ifdef SHOW_DETAILED_TIMING
#error SHOW_DETAILED_TIMING needs to be rewritten
    time_gauss->start();
    time_gauss->stop();
    time_mag->start();
    time_mag->stop();
    time_hyst->start();
    time_hyst->stop();
    time_thin->start();
    time_thin->stop();
    time_desc->start();
    time_desc->stop();
    time_vote->start();
    time_vote->stop();
#endif // not SHOW_DETAILED_TIMING

    _frame[i]->applyGauss( _params ); // async
    _frame[i]->applyMag();  // async

    _frame[i]->applyHyst();  // async
    _frame[i]->applyThinning();  // async

    _frame[i]->applyDesc();  // async
    _frame[i]->applyVoteConstructLine();  // async
    _frame[i]->applyVoteSortUniq();  // async

    _frame[i]->applyPlaneDownload(); // async
    _frame[i]->applyGaussDownload(); // async
    _frame[i]->applyMagDownload();
    _frame[i]->applyThinDownload(); // sync

    _frame[i]->applyVoteEval();  // async
    _frame[i]->applyVoteIf();  // async

    _frame[i]->applyVoteDownload();   // sync!

    cudaStreamSynchronize( _frame[i]->_stream );
    cudaStreamSynchronize( _frame[i]->_download_stream );


#ifdef SHOW_DETAILED_TIMING
    time_gauss->report( "time for Gauss " );
    time_mag  ->report( "time for Mag   " );
    time_hyst ->report( "time for Hyst  " );
    time_thin ->report( "time for Thin  " );
    time_desc ->report( "time for Desc  " );
    time_vote ->report( "time for Vote  " );
    delete time_gauss;
    delete time_mag;
    delete time_hyst;
    delete time_thin;
    delete time_desc;
    delete time_vote;
#endif // not NDEBUG
}

__host__
void TagPipe::convertToHost( size_t                          layer,
                             std::vector<cctag::EdgePoint>&  vPoints,
                             cctag::EdgePointsImage&         edgeImage,
                             std::vector<cctag::EdgePoint*>& seeds,
                             cctag::WinnerMap&               winners )
{
    assert( layer < _frame.size() );

    _frame[layer]->applyExport( vPoints, edgeImage, seeds, winners );

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

void TagPipe::debug_cmp_edge_table( int                           layer,
                                    const cctag::EdgePointsImage& cpu,
                                    const cctag::EdgePointsImage& gpu,
                                    const cctag::Parameters&      params )
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
             << "diffcpugpu-" << layer << "-edge.ppm";

    cv::cuda::PtrStepSzb plane;
    plane.data = new uint8_t[ cpu.shape()[0] * cpu.shape()[1] ];
    plane.step = cpu.shape()[0];
    plane.cols = cpu.shape()[0];
    plane.rows = cpu.shape()[1];

    if( gpu.size() != 0 && gpu.size() != 0 ) {
        for( int y=0; y<cpu.shape()[1]; y++ ) {
            for( int x=0; x<cpu.shape()[0]; x++ ) {
                if( cpu[x][y] != 0 && gpu[x][y] == 0 )
                    plane.ptr(y)[x] = DebugImage::BLUE;
                else if( cpu[x][y] == 0 && gpu[x][y] != 0 )
                    plane.ptr(y)[x] = DebugImage::GREEN;
                else if( cpu[x][y] != 0 && gpu[x][y] != 0 )
                    plane.ptr(y)[x] = DebugImage::GREY1;
                else
                    plane.ptr(y)[x] = DebugImage::BLACK;
            }
        }

        DebugImage::writePPM( filename.str(), plane );
    }

    delete [] plane.data;
}

__host__
bool TagPipe::imageCenterOptLoop(
    int                                        level,
    const cctag::numerical::geometry::Ellipse& ellipse,
    cctag::Point2dN<double>&                   center,
    const std::vector<cctag::ImageCut>&        vCuts,
    cctag::numerical::BoundedMatrix3x3d&       bestHomographyOut,
    const cctag::Parameters&                   params,
    NearbyPoint*                               cctag_pointer_buffer )
{
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

    bool success = _frame[level]->imageCenterOptLoop( e,
                                                      f,
                                                      vCuts,
                                                      bestHomography,
                                                      params,
                                                      cctag_pointer_buffer );

    if( success ) {
        center.x() = f.x;
        center.y() = f.y;

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


}; // namespace popart

