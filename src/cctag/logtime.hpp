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
            : _probe( 0 )
        { }

        void log( const char* probename, const btime::time_duration& duration ) {
            if( not _probe ) _probe = strdup( probename );
            _ms_acc( duration.total_milliseconds() );
            _us_acc( duration.total_microseconds() );
        }

        bool doPrint( ) const {
            return ( _probe != 0 );
        }

        void print( std::ostream& ostr ) const {
            if( not _probe ) return;
            ostr << _probe << ": "
                 << bacc::mean(_ms_acc) << "ms "
                 // << bacc::mean(_us_acc) << "us"
                 << std::endl;
        }

    private:
        const char* _probe;
        bacc::accumulator_set<long, bacc::features<bacc::tag::mean> > _ms_acc;
        bacc::accumulator_set<long, bacc::features<bacc::tag::mean> > _us_acc;
    };

    btime::ptime             _previous_time;
    std::vector<Measurement> _durations;
    int                      _reserved;
    int                      _idx;

    Mgmt( int rsvp )
        : _previous_time( btime::microsec_clock::local_time() )
        , _durations( rsvp )
        , _reserved( rsvp )
        , _idx( 0 )
    { }

    void resetStartTime( )
    {
        _previous_time = btime::microsec_clock::local_time();
        _idx = 0;
    }

    void log( const char* probename ) {
// std::cerr << "logging >>>" << probename << "<<<" << std::endl;
        if( _idx >= _reserved ) return;

        btime::ptime now( btime::microsec_clock::local_time() );
        btime::time_duration duration = now - _previous_time;
        _previous_time = now;
        _durations[_idx].log( probename, duration );
        _idx++;
    }

    void print( std::ostream& ostr ) const {
        int idx = 0;
        for( const Measurement& m : _durations ) {
            if( m.doPrint() ) {
                ostr << "(" << idx++ << ") ";
                m.print( ostr );
            }
        }
    }
};

} // logtime
} // cctag

