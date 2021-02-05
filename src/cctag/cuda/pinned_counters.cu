/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "pinned_counters.h"
#include "debug_macros.hpp"
#include "onoff.h"

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <iostream>

namespace cctag {

using namespace std;

bool PinnedCounters::_max_values_set = false;
int  PinnedCounters::_max_counters   = 100;
int  PinnedCounters::_max_points     = MAX_MARKER_FOR_IDENT;

/* This is system-wide unique allocation
 */
PinnedCounters pinned_counters[MAX_PIPES];

PinnedCounters::PinnedCounters( )
    : _counters( 0 )
    , _allocated_counters( 0 )
    , _nearby_points( 0 )
    , _nearby_point_counter( 0 )
{ }

PinnedCounters::~PinnedCounters( )
{
}

void PinnedCounters::setGlobalMax( int max_counters, int max_points )
{
    if( _max_values_set )
    {
        return;
    }

    _max_counters   = max_counters;
    _max_points     = max_points;
    _max_values_set = true;
}

void PinnedCounters::init( int tagPipe )
{
    _max_values_set = true;

    pinned_counters[tagPipe].obj_init( );
}

void PinnedCounters::release( int tagPipe )
{
    POP_CUDA_FREE_HOST( pinned_counters[tagPipe]._counters );
    POP_CUDA_FREE_HOST( pinned_counters[tagPipe]._nearby_points );
}

int& PinnedCounters::getCounter( int tagPipe )
{
    return pinned_counters[tagPipe].obj_getCounter( );
}

NearbyPoint& PinnedCounters::getPoint( int tagPipe, const char* file, int line )
{
    return pinned_counters[tagPipe].obj_getPoint( file, line );
}

NearbyPoint* PinnedCounters::getPointPtr( int tagPipe, const char* file, int line )
{
    return pinned_counters[tagPipe].obj_getPointPtr( file, line );
}

void PinnedCounters::obj_init( )
{
    _max_values_set = true;

    _lock.lock();
    if( ! _counters ) {
        POP_CUDA_MALLOC_HOST( &_counters, _max_counters*sizeof(int) );
    }
    if( ! _nearby_points ) {
        POP_CUDA_MALLOC_HOST( &_nearby_points, _max_points*sizeof(NearbyPoint) );
    }
    _lock.unlock();
}

int& PinnedCounters::obj_getCounter( )
{
    _lock.lock();
    if( _allocated_counters < _max_counters ) {
        int idx = _allocated_counters++;
        _lock.unlock();
        return _counters[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    Hard-coded number of integer counters in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

NearbyPoint& PinnedCounters::obj_getPoint( const char* file, int line )
{
    _lock.lock();
    if( _nearby_point_counter < _max_points ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return _nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    called from " << file << ":" << line
             << "    Hard-coded number of Nearby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

NearbyPoint* PinnedCounters::obj_getPointPtr( const char* file, int line )
{
    _lock.lock();
    if( _nearby_point_counter < _max_points ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return &_nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    called from " << file << ":" << line
             << "    Hard-coded number of Nearyby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        return 0;
    }
}

void PinnedCounters::releaseAllPoints( int tagPipe )
{
    pinned_counters[tagPipe]._lock.lock();
    pinned_counters[tagPipe]._nearby_point_counter = 0;
    pinned_counters[tagPipe]._lock.unlock();
}

} // namespace cctag

