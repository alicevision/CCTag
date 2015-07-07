#pragma once

#include <list>
#include <cuda_runtime.h>

using namespace std;

namespace popart {

struct KeepTime
{
    cudaStream_t _stream;
    cudaEvent_t  _start, _stop;

    list<cudaEvent_t> _other_events;

    KeepTime( cudaStream_t stream );
    ~KeepTime( );

    void start();
    void stop( );
    void report( const char* msg );

    float getElapsed( ); // very careful with this one, it creates synchrony

    void waitFor( cudaStream_t otherStream );
};

} // namespace popart

