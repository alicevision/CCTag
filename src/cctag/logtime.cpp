#include "logtime.hpp"

namespace cctag {
namespace logtime {

bool Mgmt::Measurement::doPrint( ) const
{
    return ( _probe != 0 );
}

void Mgmt::Measurement::print( std::ostream& ostr ) const
{
    if( not _probe ) return;
    ostr << _probe << ": "
         << bacc::mean(_ms_acc) << "ms "
         // << bacc::mean(_us_acc) << "us"
         << std::endl;
}

Mgmt::Mgmt( int rsvp )
    : _previous_time( btime::microsec_clock::local_time() )
    , _durations( rsvp )
    , _reserved( rsvp )
    , _idx( 0 )
{ }

void Mgmt::resetStartTime( )
{
    _previous_time = btime::microsec_clock::local_time();
    _idx = 0;
}

void Mgmt::print( std::ostream& ostr ) const
{
    int idx = 0;
    for( const Measurement& m : _durations ) {
        if( m.doPrint() ) {
	    ostr << "(" << idx++ << ") ";
	    m.print( ostr );
        }
    }
}

} // logtime
} // cctag

