/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <list>
#include <cctag/cuda/cctag_cuda_runtime.h>

using namespace std;

namespace cctag {

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

} // namespace cctag

