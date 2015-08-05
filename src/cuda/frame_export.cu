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
    int sz = _vote._chained_edgecoords.host.size;

    if( sz <= 0 ) return false;

    edgesMap.resize( boost::extents[ _d_plane.cols ][ _d_plane.rows ] );
    std::fill( edgesMap.origin(), edgesMap.origin() + edgesMap.size(), (cctag::EdgePoint*)NULL );

    cctag::EdgePoint* array = new cctag::EdgePoint[ sz ];
    for( int i=0; i<sz; i++ ) {
        const TriplePoint& pt = _vote._chained_edgecoords.host.ptr[i];

        array[i].init( pt.coord.x, pt.coord.y, pt.d.x, pt.d.y );
        array[i]._flowLength = pt._flowLength;
        array[i]._isMax      = pt._winnerSize;

        edgesMap[pt.coord.x][pt.coord.y] = &array[i];
    }
    for( int i=0; i<sz; i++ ) {
        cout << "  " << array[i] << " FL=" << array[i]._flowLength
             << " VT=" << array[i]._isMax
             << endl;
    }
    return true;
}

}; // namespace popart

