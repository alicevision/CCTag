#include "pinned_counters.h"
#include "debug_macros.hpp"

#include <cuda_runtime.h>
#include <iostream>

namespace popart {

using namespace std;

const int PinnedCounters::_max_counters = 100;
const int PinnedCounters::_max_points   = 30;

/* This is system-wide unique allocation
 */
PinnedCounters pinned_counters;

PinnedCounters::PinnedCounters( )
    : _counters( 0 )
    , _allocated_counters( 0 )
    , _nearby_points( 0 )
    , _nearby_point_counter( 0 )
{ }

PinnedCounters::~PinnedCounters( )
{
}

void PinnedCounters::init( ) {
    pinned_counters.obj_init();
}

void PinnedCounters::release( ) {
    POP_CUDA_FREE_HOST( pinned_counters._counters );
    POP_CUDA_FREE_HOST( pinned_counters._nearby_points );
}

int& PinnedCounters::getCounter( )
{
    return pinned_counters.obj_getCounter( );
}

NearbyPoint& PinnedCounters::getPoint( )
{
    return pinned_counters.obj_getPoint( );
}

NearbyPoint* PinnedCounters::getPointPtr( )
{
    return pinned_counters.obj_getPointPtr( );
}

void PinnedCounters::obj_init( )
{
    _lock.lock();
    if( not _counters ) {
        cudaError_t err;
        err = cudaMallocHost( &_counters, _max_counters*sizeof(int) );

        POP_CUDA_FATAL_TEST( err, "Could not allocate global int counters: " );
    }
    if( not _nearby_points ) {
        cudaError_t err;
        err = cudaMallocHost( &_nearby_points, _max_points*sizeof(NearbyPoint) );

        POP_CUDA_FATAL_TEST( err, "Could not allocate global nearby point structs: " );
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

NearbyPoint& PinnedCounters::obj_getPoint( )
{
    _lock.lock();
    if( _nearby_point_counter < _max_points ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return _nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    Hard-coded number of Nearyby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

NearbyPoint* PinnedCounters::obj_getPointPtr( )
{
    _lock.lock();
    if( _nearby_point_counter < _max_points ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return &_nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    Hard-coded number of Nearyby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

void PinnedCounters::releaseAllPoints( )
{
    pinned_counters._lock.lock();
    pinned_counters._nearby_point_counter = 0;
    pinned_counters._lock.unlock();
}

} // namespace popart

