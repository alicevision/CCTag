#include "tag.h"
#include "frame.h"
#include "debug_macros.hpp"
#include "keep_time.hpp"
#include <sstream>
#include <iostream>

using namespace std;

namespace popart
{

__host__
TagPipe::TagPipe( )
{
    for( int i=0; i<4; i++ ) _frame[i] = 0;
}

__host__
void TagPipe::prepframe( const uint32_t pix_w, const uint32_t pix_h,
                         const cctag::Parameters& params )
{
    cerr << "Enter " << __FUNCTION__ << endl;
    static bool tables_initialized = false;
    if( not tables_initialized ) {
        tables_initialized = true;
        Frame::initGaussTable( );
        Frame::initThinningTable( );
    }

    uint32_t w = pix_w;
    uint32_t h = pix_h;
    for( int i=0; i<4; i++ ) {
        _frame[i] = new popart::Frame( w, h ); // sync
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }

    _frame[0]->createTexture( popart::FrameTexture::normalized_uchar_to_float); // sync
    _frame[0]->allocUploadEvent( ); // sync

    for( int i=0; i<4; i++ ) {
        _frame[i]->allocDevGaussianPlane( params ); // sync
        _frame[i]->allocDoneEvent( ); // sync
    }
    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void TagPipe::tagframe( unsigned char* pix,
                        const uint32_t pix_w,
                        const uint32_t pix_h,
                        const cctag::Parameters& params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

#if 0
    _frame[0]->upload( pix ); // async

    FrameEvent ev = _frame[0]->addUploadEvent( ); // async

    for( int i=1; i<4; i++ ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
        // _frame[i]->fillFromFrame( *(_frame[0]) );
    }

    for( int i=0; i<4; i++ ) {
        _frame[i]->applyGauss( params ); // async
    }

    KeepTime t( _frame[0]->_stream );
    t.start();

    for( int i=0; i<4; i++ ) {
        _frame[i]->applyMag(   params );  // async
        _frame[i]->applyHyst(  params );  // async
        _frame[i]->applyMore(  params );  // async
        _frame[i]->applyVote(  params );  // async
    }

    FrameEvent doneEv[4];
    for( int i=1; i<4; i++ ) {
        doneEv[i] = _frame[i]->addDoneEvent( ); // async
    }
    for( int i=1; i<4; i++ ) {
        _frame[0]->streamSync( doneEv[i] ); // aysnc
    }
    t.stop();
    t.report( "Time for all frames " );
#else
    KeepTime t( _frame[0]->_stream );
    t.start();

    _frame[0]->upload( pix ); // async

    /*
     * First thing to do seems to be Canny
     *  tracked to cannyCv in canny.tcc
     *   going to cvCanny( 3-layer out, 1 layer in, low thresh, high thresh )
     *   tracked to cvCanny in canny.cpp
     *    going to cvRecodedCanny( 1 layer in, 1 8-bit layer out, 1 16-bit layer out (dx), 1 16-bit layer out (dy), low thresh * 256, high thresh * 256, aperture size "3", CV_CANNY_L2_GRADIENT )
     *    tracked to cvRecodedCanny in filter/cvRecode.cpp
     *     first: apply interesting sequence of Gaussian filters
     *    ...
     *  on return, convert all 3 out layers to 3-layer out; implicit typecast 16-to-8-bits
     */

    FrameEvent ev = _frame[0]->addUploadEvent( ); // async

    for( int i=1; i<4; i++ ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
        // _frame[i]->fillFromFrame( *(_frame[0]) );
    }

    for( int i=0; i<4; i++ ) {
        _frame[i]->applyGauss( params ); // async
        _frame[i]->applyMag(   params );  // async
        _frame[i]->applyHyst(  params );  // async
        _frame[i]->applyMore(  params );  // async
        _frame[i]->applyVote(  params );  // async
    }

    FrameEvent doneEv[4];
    for( int i=1; i<4; i++ ) {
        doneEv[i] = _frame[i]->addDoneEvent( ); // async
    }
    for( int i=1; i<4; i++ ) {
        _frame[0]->streamSync( doneEv[i] ); // aysnc
    }
    t.stop();
    t.report( "Time for all frames " );
#endif
    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void TagPipe::debug( unsigned char* pix, const cctag::Parameters& params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    if( true ) {
        // This is a debug block

        for( int i=0; i<4; i++ ) {
            _frame[i]->hostDebugDownload( params );
        }
        POP_SYNC_CHK;

        _frame[0]->hostDebugCompare( pix );

        for( int i=0; i<4; i++ ) {
            std::ostringstream ostr;
            ostr << "debug-input-plane-" << i;
            _frame[i]->writeHostDebugPlane( ostr.str(), params );
        }
        POP_SYNC_CHK;
    }

    cerr << "terminating in tagframe" << endl;
    cerr << "Leave " << __FUNCTION__ << endl;
    exit( 0 );
}

}; // namespace popart

