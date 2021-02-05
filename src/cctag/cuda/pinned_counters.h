/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "cctag/cuda/nearby_point.h"

#include <mutex>

namespace cctag {

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

    static void setGlobalMax( int max_counters, int max_points );

    static void init( int tagPipe );
    static void release( int tagPipe );

    static int& getCounter( int tagPipe );

    /** Returns a reference to a NearbyPoint-sized section of host-side
     *  pinned memory.
     *  This function is only used by the constructors of the class
     *  CCTag in cctag before identification.
     */
    static NearbyPoint& getPoint( int tagPipe, const char* file, int line );
    static NearbyPoint* getPointPtr( int tagPipe, const char* file, int line );

    /** Called after all identification of all CCTags is complete.
     *  Invalidates all NearbyPoint references in all CCTag.
     */
    static void releaseAllPoints( int tagPipe );

private:
    int*         _counters;
    int          _allocated_counters;
    NearbyPoint* _nearby_points;
    int          _nearby_point_counter;

    std::mutex _lock;

    static const bool _max_values_set;
    static const int  _max_counters;
    static const int  _max_points;

    void         obj_init( );
    int&         obj_getCounter( );
    NearbyPoint& obj_getPoint( const char* file, int line );
    NearbyPoint* obj_getPointPtr( const char* file, int line );
};

} // namespace cctag

