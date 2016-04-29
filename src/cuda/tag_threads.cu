#include <iostream>

#include "cuda/tag_threads.h"
#include "cuda/tag.h"
#include "cuda/frame.h"

namespace popart
{

using namespace std;

TagThread::TagThread( TagThreads* creator, TagPipe* pipe, int layer )
    : boost::thread( &TagThread::call, this )
    , _creator( creator )
    , _pipe( pipe )
    , _my_layer( layer )
{ }

#define FORCE_SERIALIZE_LAYERS

#ifdef FORCE_SERIALIZE_LAYERS
static boost::mutex serialize;
#endif

void TagThread::call( void )
{
    _creator->startWait( );

    while( true ) {
        _creator->frameReadyWait( );

#ifdef FORCE_SERIALIZE_LAYERS
        serialize.lock();
#endif
        _pipe->handleframe( _my_layer );
#ifdef FORCE_SERIALIZE_LAYERS
        serialize.unlock();
#endif

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

}; // namespace popart

