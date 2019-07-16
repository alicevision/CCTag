/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "LogTime.hpp"

namespace cctag {
namespace logtime {

bool Mgmt::Measurement::doPrint( ) const
{
    return ( _probe != nullptr );
}

void Mgmt::Measurement::print( std::ostream& ostr ) const
{
    if( ! _probe ) return;
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

