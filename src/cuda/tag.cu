#include "tag.h"
#include "frame.h"
#include "debug_macros.hpp"
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
    }
    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void TagPipe::tagframe( unsigned char* pix, const uint32_t pix_w, const uint32_t pix_h )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    _frame[0]->upload( pix ); // async

    FrameEvent ev = _frame[0]->addUploadEvent( ); // async

    for( int i=1; i<4; i++ ) {
        _frame[i]->streamSync( ev ); // aysnc
        _frame[i]->fillFromTexture( *(_frame[0]) ); // aysnc
        // _frame[i]->fillFromFrame( *(_frame[0]) );
    }

    for( int i=0; i<4; i++ ) {
        _frame[i]->applyGauss(); // async
    }
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

