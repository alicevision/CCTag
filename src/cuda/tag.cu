#include "tag.h"
#include "frame.h"
#include <sstream>
#include <iostream>

using namespace std;

namespace popart
{

__host__
void tagframe( unsigned char* pix, uint32_t pix_w, uint32_t pix_h )
{
    unsigned char* verify = new unsigned char[pix_w * pix_h];
    memset( verify, 0, pix_w * pix_h );

    // popart::Frame::writeDebugPlane( "debug-input-base.pgm", pix, pix_w, pix_h );

    popart::Frame* frame[4];
    uint32_t w = pix_w;
    uint32_t h = pix_h;
    for( int i=0; i<4; i++ ) {
        frame[i] = new popart::Frame( sizeof(unsigned char), w, h );
        w = ( w >> 1 ) + ( w & 1 );
        h = ( h >> 1 ) + ( h & 1 );
    }
    frame[0]->upload( pix );
    frame[0]->download( verify, pix_w, pix_h );
    frame[0]->createTexture( popart::FrameTexture::normalized_uchar_to_float);

    frame[0]->streamSync( );
    for( int i=1; i<4; i++ ) {
        // frame[i]->fillFromFrame( *(frame[0]) );
        frame[i]->fillFromTexture( *(frame[0]) );
    }
    for( int i=1; i<4; i++ ) {
        frame[i]->streamSync( );
    }
    for( int i=0; i<4; i++ ) {
        frame[i]->hostDebugDownload();
    }
    for( int i=0; i<4; i++ ) {
        std::ostringstream ostr;
        ostr << "debug-input-plane-" << i << ".pgm";
        frame[i]->writeHostDebugPlane( ostr.str().c_str() );
    }
    cudaDeviceSynchronize();
    cerr << "terminating in tagframe" << endl;
    exit( 0 );
}

}; // namespace popart

