#pragma once

#include <boost/thread/mutex.hpp>
#include "cuda/nearby_point.h"

namespace popart {

/** Pinned Counters provides other classes with pieces of memory that
 *  have been allocated in pinned memory.
 *  Since pinned memory can only be allocated on page boundaries, and
 *  it steals part of physical memory, it is very expensive to waste it.
 *  The CCTag port requires the transfer of many tiny counters between
 *  host and device, and these transfers are only asynchronous if the
 *  host side is pinned.
 *  This class provides the cheaper, but fairly dangerous means for
 *  doing that.
 *
 *  This is not a memory allocation, but works like the ancient
 *  Unix sbrk(). This means that you can never release individual
 *  structure. You can only release all of these points at once.
 */
class PinnedCounters
{
public:
    PinnedCounters( );
    ~PinnedCounters( );

    static void init( );
    static void release( );

    static int& getCounter( );

    /** Returns a reference to a NearyPoint-sized section of host-side
     *  pinned memory.
     *  This function is only used by the constructors of the class
     *  CCTag in cctag before identification.
     */
    static NearbyPoint& getPoint( );
    static NearbyPoint* getPointPtr( );

    /** Called after all identification of all CCTags is complete.
     *  Invalidates all NearbyPoint references in all CCTag.
     */
    static void releaseAllPoints( );

private:
    int*         _counters;
    int          _allocated_counters;
    NearbyPoint* _nearby_points;
    int          _nearby_point_counter;

    boost::mutex _lock;

    static const int _max_counters;
    static const int _max_points;

    void         obj_init( );
    int&         obj_getCounter( );
    NearbyPoint& obj_getPoint( );
    NearbyPoint* obj_getPointPtr( );
};

} // namespace popart

