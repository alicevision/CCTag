#include "tag.h"
#include "frame.h"
#include "debug_macros.hpp"
#include "keep_time.hpp"
#include <sstream>
#include <iostream>
#include <fstream>

#include "debug_image.h"
#include "cctag/talk.hpp"

#if 1 // #ifndef NDEBUG
#define SHOW_TIMING
#else
#undef  SHOW_TIMING
#endif

using namespace std;

namespace popart
{

__host__
void TagPipe::initialize( const uint32_t pix_w,
                          const uint32_t pix_h,
                          const cctag::Parameters& params )
{
    static bool tables_initialized = false;
    if( not tables_initialized ) {
        tables_initialized = true;
        Frame::initGaussTable( );
        Frame::initThinningTable( );
    }

    int num_layers = params._numberOfMultiresLayers;
    _frame.reserve( num_layers );

    uint32_t w = pix_w;
    uint32_t h = pix_h;
    for( int i=0; i<num_layers; i++ ) {
        _frame.push_back( new popart::Frame( w, h, i ) ); // sync
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }

    _frame[0]->createTexture( popart::FrameTexture::normalized_uchar_to_float); // sync
    _frame[0]->allocUploadEvent( ); // sync

    for( int i=0; i<num_layers; i++ ) {
        _frame[i]->allocRequiredMem( params ); // sync
        _frame[i]->allocDoneEvent( ); // sync
    }
}

__host__
void TagPipe::load( unsigned char* pix )
{
    KeepTime t( _frame[0]->_stream );
    t.start();

    _frame[0]->upload( pix ); // async

    t.stop();
    t.report( "Time for frame upload " );
}

__host__
void TagPipe::tagframe( const cctag::Parameters& params )
{
    int num_layers = _frame.size();

#ifdef SHOW_TIMING
    KeepTime* time_gauss[num_layers];
    KeepTime* time_mag  [num_layers];
    KeepTime* time_hyst [num_layers];
    KeepTime* time_thin [num_layers];
    KeepTime* time_desc [num_layers];
    KeepTime* time_vote [num_layers];
    for( int i=0; i<num_layers; i++ ) {
        time_gauss[i] = new KeepTime( _frame[i]->_stream );
        time_mag  [i] = new KeepTime( _frame[i]->_stream );
        time_hyst [i] = new KeepTime( _frame[i]->_stream );
        time_thin [i] = new KeepTime( _frame[i]->_stream );
        time_desc [i] = new KeepTime( _frame[i]->_stream );
        time_vote [i] = new KeepTime( _frame[i]->_stream );
    }
#endif // not NDEBUG

    KeepTime t( _frame[0]->_stream );
    t.start();

    for( int i=0; i<num_layers; i++ ) {
        _frame[i]->initRequiredMem( ); // async
    }

    FrameEvent ev = _frame[0]->addUploadEvent( ); // async

    for( int i=1; i<num_layers; i++ ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
        // _frame[i]->fillFromFrame( *(_frame[0]) );
    }

    for( int i=0; i<num_layers; i++ ) {
        bool success;
#ifdef SHOW_TIMING
        time_gauss[i]->start();
        _frame[i]->applyGauss( params ); // async
        time_gauss[i]->stop();
        POP_CHK_CALL_IFSYNC;
        time_mag[i]->start();
        _frame[i]->applyMag(   params );  // async
        time_mag[i]->stop();
        POP_CHK_CALL_IFSYNC;
        time_hyst[i]->start();
        _frame[i]->applyHyst(  params );  // async
        POP_CHK_CALL_IFSYNC;
        time_hyst[i]->stop();
        time_thin[i]->start();
        _frame[i]->applyThinning(  params );  // async
        time_thin[i]->stop();
        POP_CHK_CALL_IFSYNC;
        time_desc[i]->start();
        success = _frame[i]->applyDesc(  params );  // async
        time_desc[i]->stop();
        POP_CHK_CALL_IFSYNC;

        if( not success ) continue;

        time_vote[i]->start();
        _frame[i]->applyVote(  params );  // async
        time_vote[i]->stop();
        POP_CHK_CALL_IFSYNC;

        _frame[i]->applyThinDownload(  params ); // has a sync step
        POP_CHK_CALL_IFSYNC;
#else
        _frame[i]->applyGauss( params ); // async
        POP_CHK_CALL_IFSYNC;
        _frame[i]->applyMag(   params );  // async
        POP_CHK_CALL_IFSYNC;
        _frame[i]->applyHyst(  params );  // async
        POP_CHK_CALL_IFSYNC;
        _frame[i]->applyThinning(  params );  // async
        POP_CHK_CALL_IFSYNC;
        success = _frame[i]->applyDesc(  params );  // async
        POP_CHK_CALL_IFSYNC;

        if( not success ) continue;

        _frame[i]->applyVote(  params );  // async
        POP_CHK_CALL_IFSYNC;
        _frame[i]->applyThinDownload(  params ); // has a sync step
        POP_CHK_CALL_IFSYNC;
#endif // not NDEBUG

        // _frame[i]->applyLink(  params );  // async
    }

    FrameEvent doneEv[num_layers];
    for( int i=1; i<num_layers; i++ ) {
        doneEv[i] = _frame[i]->addDoneEvent( ); // async
    }
    for( int i=1; i<num_layers; i++ ) {
        _frame[0]->streamSync( doneEv[i] ); // aysnc
    }
    t.stop();
    t.report( "Time for all frames " );

#ifdef SHOW_TIMING
    for( int i=0; i<num_layers; i++ ) {
        DO_TALK(
          time_gauss[i]->report( "time for Gauss " );
          time_mag  [i]->report( "time for Mag   " );
          time_hyst [i]->report( "time for Hyst  " );
          time_thin [i]->report( "time for Thin  " );
          time_desc [i]->report( "time for Desc  " );
          time_vote [i]->report( "time for Vote  " );
        )
        delete time_gauss[i];
        delete time_mag  [i];
        delete time_hyst [i];
        delete time_thin [i];
        delete time_desc [i];
        delete time_vote [i];
    }
#endif // not NDEBUG
}

__host__
void TagPipe::download( size_t                          layer,
                        std::vector<cctag::EdgePoint>&  vPoints,
                        cctag::EdgePointsImage&         edgeImage,
                        std::vector<cctag::EdgePoint*>& seeds,
                        cctag::WinnerMap&               winners )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    assert( layer < _frame.size() );

    _frame[layer]->applyExport( vPoints, edgeImage, seeds, winners );

    // cerr << "Leave " << __FUNCTION__ << endl;
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

}; // namespace popart

