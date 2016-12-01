/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "triple_point.h"
#include <iomanip>

namespace cctag {

#ifndef NDEBUG
__host__
void TriplePoint::debug_out( std::ostream& ostr ) const
{
    ostr << "orig=" << coord.x << " " << coord.y << " "
         << "d=" << d.x << " " << d.y << " "
         << "bef=" << descending.befor.x << " " << descending.befor.y << " "
         << "aft=" << descending.after.x << " " << descending.after.y << " "
         << std::setprecision(4)
         << "winS=" << _winnerSize << " l=" << _flowLength;

    if( _coords_idx != 0 ) {
        for( int i=0; i<_coords_idx; i++ ) {
            ostr << " (" << _coords[i].x << "," << _coords[i].y << ")";
        }
    }
}

__host__
std::string TriplePoint::debug_out( ) const
{
    std::ostringstream ostr;
    debug_out( ostr );
    return ostr.str();
}
#endif

}; // namespace cctag

