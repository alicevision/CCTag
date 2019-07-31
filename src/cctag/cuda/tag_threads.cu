/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include "tag_threads.h"
#include "tag.h"
#include "frame.h"

namespace cctag
{

using namespace std;

TagThread::TagThread( TagThreads* creator, TagPipe* pipe, int layer )
    : std::thread( &TagThread::call, this )
    , _creator( creator )
    , _pipe( pipe )
    , _my_layer( layer )
{ }

void TagThread::call( void )
{
    _creator->startWait( );

    while( true ) {
        _creator->frameReadyWait( );

        _pipe->handleframe( _my_layer );

        _creator->frameDonePost( );
    }
}

TagThreads::TagThreads( )
    : _start( 0 )
    , _frameReady( 0 )
    , _frameDone( 0 )
{ }

void TagThreads::init( TagPipe* pipe, int layers )
{
    _pipe   = pipe;
    _layers = layers;

    for( int i=0; i<_layers; i++ ) {
        new TagThread( this, _pipe, i );
    }

    startPost( );
}

void TagThreads::oneRound( )
{
    frameReadyPost( );
    frameDoneWait( );
}

void TagThreads::startWait( )      { _start.wait( 1 );  }
void TagThreads::startPost( )      { _start.post( _layers ); }
void TagThreads::frameReadyWait( ) { _frameReady.wait( 1 );  }
void TagThreads::frameReadyPost( ) { _frameReady.post( _layers ); }
void TagThreads::frameDoneWait( )  { _frameDone.wait( _layers );  }
void TagThreads::frameDonePost( )  { _frameDone.post( 1 ); }

/*************************************************************
 * TagSemaphore
 *************************************************************/

void TagSemaphore::wait( int n )
{
    std::unique_lock<std::mutex> sema_lock( _sema_mx );
    while( _sema_val - n < 0 )
    {
        _sema_cond.wait( sema_lock );
    }
    _sema_val -= n;
    sema_lock.unlock();
}

void TagSemaphore::post( int n )
{
    std::unique_lock<std::mutex> sema_lock( _sema_mx );
    _sema_val += n;
    _sema_cond.notify_all();
    sema_lock.unlock();
}

}; // namespace cctag

