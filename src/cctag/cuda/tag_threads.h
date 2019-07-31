/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include <string>
// #include <vector>
// #include <stdlib.h>
// #include <inttypes.h>
// #include <opencv2/core.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>

// #include "cctag/cuda/onoff.h"
// #include "cctag/Params.hpp"
// #include "cctag/Types.hpp"
// #include "cctag/ImageCut.hpp"
// #include "cctag/geometry/Ellipse.hpp"
// #include "cctag/geometry/Point.hpp"
// #include "cctag/algebra/matrix/Matrix.hpp"
// #include "cctag/utils/LogTime.hpp"

namespace cctag
{

class TagPipe;
class TagThreads;

/*************************************************************
 * TagSemaphore
 *************************************************************/
class TagSemaphore
{
    int                          _sema_val;
    std::mutex                   _sema_mx;
    std::condition_variable      _sema_cond;
public:
    TagSemaphore( int init )
        : _sema_val( init )
    { }

    void wait( int n = 1 );
    void post( int n = 1 );
};

class TagThread : public std::thread
{
    TagThreads* _creator;
    TagPipe*    _pipe;
    int         _my_layer;

public:
    TagThread( TagThreads* creator, TagPipe* pipe, int layer );

    void call( void );
};

class TagThreads
{
    TagPipe*         _pipe;
    int              _layers;
    TagSemaphore     _start;
    TagSemaphore     _frameReady;
    TagSemaphore     _frameDone;
public:
    TagThreads( );

    void init( TagPipe* pipe, int layers );

    void oneRound( );

    void startWait( );
    void startPost( );
    void frameReadyWait( );
    void frameReadyPost( );
    void frameDoneWait( );
    void frameDonePost( );
};

}; // namespace cctag

