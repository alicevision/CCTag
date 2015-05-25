#include "tag.h"
#include "frame.h"
#include "debug_macros.hpp"
#include <sstream>
#include <iostream>

using namespace std;

namespace popart
{

__host__
void tagframe( unsigned char* pix, const uint32_t pix_w, const uint32_t pix_h )
{
    cerr << "Enter " << __FUNCTION__ << endl;
    static bool gauss_table_initialized = false;
    if( not gauss_table_initialized ) {
        Frame::initGaussTable( );
    }

    unsigned char* verify = new unsigned char[pix_w * pix_h];
    memset( verify, 0, pix_w * pix_h );

    popart::Frame* frame[4];
    uint32_t w = pix_w;
    uint32_t h = pix_h;
    for( int i=0; i<4; i++ ) {
        frame[i] = new popart::Frame( w, h ); // sync
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
    frame[0]->upload( pix ); // async
    frame[0]->createTexture( popart::FrameTexture::normalized_uchar_to_float); // sync

    frame[0]->streamSync( );
    POP_SYNC_CHK;
    for( int i=1; i<4; i++ ) {
        // frame[i]->fillFromFrame( *(frame[0]) );
        frame[i]->fillFromTexture( *(frame[0]) );
    }

    for( int i=0; i<4; i++ ) {
        frame[i]->allocDevGaussianPlane();
        frame[i]->applyGauss();
    }

    if( true ) {
        // This is a debug block

        for( int i=0; i<4; i++ ) {
            frame[i]->hostDebugDownload();
        }
        POP_SYNC_CHK;

        frame[0]->hostDebugCompare( pix );

        for( int i=0; i<4; i++ ) {
            std::ostringstream ostr;
            ostr << "debug-input-plane-" << i;
            frame[i]->writeHostDebugPlane( ostr.str() );
        }
        POP_SYNC_CHK;
    }

    cerr << "terminating in tagframe" << endl;
    cerr << "Leave " << __FUNCTION__ << endl;
    exit( 0 );
}

}; // namespace popart

