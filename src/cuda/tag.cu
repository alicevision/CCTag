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
void TagPipe::prepframe( const uint32_t pix_w, const uint32_t pix_h )
{
    cerr << "Enter " << __FUNCTION__ << endl;
    static bool gauss_table_initialized = false;
    if( not gauss_table_initialized ) {
        Frame::initGaussTable( );
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
        _frame[i]->allocDevGaussianPlane(); // sync
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
    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void TagPipe::debug( unsigned char* pix )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    if( true ) {
        // This is a debug block

        for( int i=0; i<4; i++ ) {
            _frame[i]->hostDebugDownload();
        }
        POP_SYNC_CHK;

        _frame[0]->hostDebugCompare( pix );

        for( int i=0; i<4; i++ ) {
            std::ostringstream ostr;
            ostr << "debug-input-plane-" << i;
            _frame[i]->writeHostDebugPlane( ostr.str() );
        }
        POP_SYNC_CHK;
    }

    cerr << "terminating in tagframe" << endl;
    cerr << "Leave " << __FUNCTION__ << endl;
    exit( 0 );
}

}; // namespace popart

