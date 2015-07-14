#pragma once

#include <cuda_runtime.h>
#include <string>
#include <sstream>

namespace popart {

/*************************************************************
 * TriplePoint
 * A simplified version of EdgePoint in the C++ code.
 *************************************************************/
struct TriplePoint
{
    int2  coord;
    int2  befor;
    int2  after;

    // in the original code, chosen keeps list of voters
    // no possible here; we must invert this
    int   my_vote;
    float chosen_flow_length;

    int   _winnerSize;
    float _flowLength;
#ifndef NDEBUG
    int   _coords_collect_idx;
    int   _coords_idx;
    int2  _coords[12];

    __device__
    inline void debug_init( ) {
        _coords_idx         = 0;
        _coords_collect_idx = 0;
    }
    __device__
    inline void debug_add( int2 c ) {
        if( _coords_collect_idx >= 12 ) return;
        _coords[_coords_collect_idx].x = c.x;
        _coords[_coords_collect_idx].y = c.y;
        _coords_collect_idx++;
    }
    __device__
    inline void debug_commit( ) {
        _coords_idx = _coords_collect_idx;
    }

    __host__
    inline std::string debug_out( ) const {
        std::ostringstream ostr;
        ostr << "orig=" << coord.x << " " << coord.y << " "
             << "bef=" << befor.x << " " << befor.y << " "
             << "aft=" << after.x << " " << after.y;
        if( _coords_idx != 0 ) {
            for( int i=0; i<_coords_idx; i++ ) {
                ostr << " (" << _coords[i].x << "," << _coords[i].y << ")";
            }
        }
        return ostr.str();

    }
#endif
};

}; // namespace popart

