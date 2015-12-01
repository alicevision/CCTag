#pragma once

#include <boost/thread/mutex.hpp>

namespace popart {

class PinnedCounters
{
public:
    PinnedCounters( );
    ~PinnedCounters( );

    void init( );

    int& getCounter( );

private:
    int*         _counters;
    int          _allocated_counters;
    boost::mutex _lock;

    static const int _max_counters;
};

extern PinnedCounters pinned_counters;

} // namespace popart

