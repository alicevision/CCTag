// #include <iostream>
// #include <limits>
// #include <assert.h>
// #include <fstream>
// #include <string.h>
#include <cuda_runtime.h>
// #include "debug_macros.hpp"

#include "frame.h"
// #include "clamp.h"
// #include "frame_gaussian.h"

namespace popart {

using namespace std;

bool Frame::applyExport( cctag::EdgePointsImage& edgesMap )
{
    int sz = _vote._all_edgecoords.host.size;

    if( sz <= 0 ) return false;

    edgesMap.resize( boost::extents[ _d_plane.cols ][ _d_plane.rows ] );
    std::fill( edgesMap.origin(), edgesMap.origin() + edgesMap.size(), (cctag::EdgePoint*)NULL );

    cctag::EdgePoint* array = new cctag::EdgePoint[ sz ];
    for( int i=0; i<sz; i++ ) {
        const int2& coord = _vote._all_edgecoords.host.ptr[i];
        int16_t     dx    = _h_dx.ptr(coord.y)[coord.x];
        int16_t     dy    = _h_dy.ptr(coord.y)[coord.x];

        assert( dx != 0 || dy != 0 );

        array[i].init( coord.x, coord.y, dx, dy );

        edgesMap[coord.x][coord.y] = &array[i];
    }
    // for( int i=0; i<sz; i++ ) { cout << "  " << array[i] << endl; }
    return true;
}

}; // namespace popart

