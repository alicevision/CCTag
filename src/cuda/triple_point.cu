#include "triple_point.h"
#include <iomanip>

namespace popart {

#ifndef NDEBUG
__host__
void TriplePoint::debug_out( std::ostream& ostr ) const
{
    ostr << "orig=" << coord.x << " " << coord.y << " "
         << "d=" << d.x << " " << d.y << " "
         << "bef=" << descending.befor.x << " " << descending.befor.y << " "
         << "aft=" << descending.after.x << " " << descending.after.y << " "
         << "vote=" << "my_vote" << " l=" << chosen_flow_length << " "
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

}; // namespace popart

