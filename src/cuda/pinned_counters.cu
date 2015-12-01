#include "pinned_counters.h"
#include "debug_macros.hpp"

#include <cuda_runtime.h>
#include <iostream>

namespace popart {

using namespace std;

const int PinnedCounters::_max_counters = 100;

PinnedCounters pinned_counters;

PinnedCounters::PinnedCounters( )
    : _counters( 0 )
    , _allocated_counters( 0 )
{ }

void PinnedCounters::init( )
{
    _lock.lock();
    if( not _counters ) {
        cudaError_t err;
        err = cudaMallocHost( &_counters, _max_counters*sizeof(int) );

        POP_CUDA_FATAL_TEST( err, "Could not allocate global int counters: " );

        _allocated_counters = 0;
    }
    _lock.unlock();
}

PinnedCounters::~PinnedCounters( )
{
    cudaFreeHost( _counters );
}

int& PinnedCounters::getCounter( )
{
    _lock.lock();
    if( _allocated_counters < _max_counters ) {
        int idx = _allocated_counters++;
        _lock.unlock();
        return _counters[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    Hard-coded number of integer counters is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

} // namespace popart

