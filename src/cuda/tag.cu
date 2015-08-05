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
        _frame[i]->allocRequiredMem( params ); // sync
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

    for( int i=0; i<4; i++ ) {
        _frame[i]->initRequiredMem( ); // async
    }

    _frame[0]->upload( pix ); // async

    FrameEvent ev = _frame[0]->addUploadEvent( ); // async

    for( int i=1; i<4; i++ ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
        // _frame[i]->fillFromFrame( *(_frame[0]) );
    }

    for( int i=0; i<4; i++ ) {
        bool success;
        _frame[i]->applyGauss( params ); // async
        _frame[i]->applyMag(   params );  // async
        _frame[i]->applyHyst(  params );  // async
        _frame[i]->applyThinning(  params );  // async
        success = _frame[i]->applyDesc(  params );  // async
        if( not success ) continue;
        _frame[i]->applyVote(  params );  // async
        // _frame[i]->applyLink(  params );  // async
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

    {
        for( int i=0; i<4; i++ ) {
            cctag::EdgePointsImage         edgeImage;
            std::vector<cctag::EdgePoint*> seeds;
            cctag::WinnerMap               winners;

            cout << "Exporting image frame " << i << endl;
            _frame[i]->applyExport( edgeImage, seeds, winners );
        }
    }

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

