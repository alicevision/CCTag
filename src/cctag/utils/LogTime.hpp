/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace cctag {
namespace logtime {

namespace btime = boost::posix_time;
namespace bacc  = boost::accumulators;

struct Mgmt
{
    class Measurement
    {
    public:
        Measurement( )
            : _probe( nullptr )
        { }

        void log( const char* probename, const btime::time_duration& duration ) {
            if( ! _probe ) _probe = strdup( probename );
            _ms_acc( duration.total_milliseconds() );
            _us_acc( duration.total_microseconds() );
        }

        bool doPrint( ) const;

        void print( std::ostream& ostr ) const;

    private:
        const char* _probe;
        bacc::accumulator_set<long, bacc::features<bacc::tag::mean> > _ms_acc;
        bacc::accumulator_set<long, bacc::features<bacc::tag::mean> > _us_acc;
    };

    btime::ptime             _previous_time;
    std::vector<Measurement> _durations;
    int                      _reserved;
    int                      _idx;

    explicit Mgmt( int rsvp );

    void resetStartTime( );

    void log( const char* probename ) {
        // std::cerr << "logging >>>" << probename << "<<<" << std::endl;
        if( _idx >= _reserved ) return;

        btime::ptime now( btime::microsec_clock::local_time() );
        btime::time_duration duration = now - _previous_time;
        _previous_time = now;
        _durations[_idx].log( probename, duration );
        _idx++;
    }

    void print( std::ostream& ostr ) const;
};

} // logtime
} // cctag

