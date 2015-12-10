#include "cctag/package.h"
#include "cctag/packagePool.h"

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/cmdline.hpp"
#include "cuda/framepackage.h"

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <stdlib.h>
#include <iostream>
// #include <sys/mman.h>

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

namespace popart
{

/* This is not a joke.
 * We need a global lock for the CUDA resources because
 * we require all memory.
 */
boost::mutex Package::_lock_phase_1;

Package::Package( PackagePool* pool,
                  const cctag::Parameters & params,
                  const cctag::CCTagMarkersBank & bank )
    : _my_pool( pool )
    , _src( 0 )
    , _params( params )
    , _bank( bank )
    , _framePackage( 0 )
    , _readyForDetection( false )
    , _detect_thread( &Package::detectionThread, this )
{
    _numLayers = params._numberOfProcessedMultiresLayers;

    winners = new WinnerMap[ _numLayers ];
    seeds   = new std::vector<EdgePoint*>[ _numLayers ];
}

void Package::resetTables( )
{
    for( int i=0; i<_numLayers; i++ ) {
        winners[i].clear();
        seeds  [i].clear();
    }
}

void Package::init( int            frameId,
                    const cv::Mat* src,
                    std::ostream*  output,
                    std::string    debugFileName )
{
    if( _src == 0 ) {
        // allocate memories
    }

    _frameId = frameId;
    _src     = src;

    if( _framePackage == 0 ) {
        _framePackage = new FramePackage* [ _numLayers ];

        uint32_t w = src->size().width;
        uint32_t h = src->size().height;

        for( int i=0; i<_numLayers; i++ ) {
            _framePackage[i] = new FramePackage( w, h );
            w = ( w >> 1 ) + ( w & 1 );
            h = ( h >> 1 ) + ( h & 1 );
        }
    }

    if (debugFileName == "") {
      debugFileName = "00000";
    }

    _output        = output;
    _debugFileName = debugFileName;

    _markers.clear();
}

void Package::lockPhase1( )
{
    _lock_phase_1.lock();

    for( int i=0; i<_numLayers; i++ ) {
        _framePackage[i]->pinAll( );
    }
}

void Package::unlockPhase1( )
{
    for( int i=0; i<_numLayers; i++ ) {
        _framePackage[i]->unpinAll( );
    }

    _lock_phase_1.unlock();
}

void Package::detect( )
{
    _rfd_lock.lock();
    _readyForDetection = true;
    _rfd_cond.notify_all( );
    _rfd_lock.unlock( );
}

void Package::detectionThread( )
{
    while( true ) {
        _rfd_lock.lock();
        while( _readyForDetection == false ) {
            _rfd_cond.wait( _rfd_lock );
        }
        _rfd_lock.unlock( );

        detectionRun( );

        _rfd_lock.lock();
        _readyForDetection = false;
        _rfd_lock.unlock( );

        _my_pool->returnPackage( this );
    }
}

void Package::detectionRun( )
{
    // Process markers detection
    boost::timer t;

    CCTagVisualDebug::instance().initBackgroundImage( *_src );
    CCTagVisualDebug::instance().setImageFileName(_debugFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

    static cctag::logtime::Mgmt* durations = 0;
#if 0
    if( not durations ) {
        durations = new cctag::logtime::Mgmt( 25 );
    } else {
        durations->resetStartTime();
    }
#endif
    cctagDetection( this, _markers, _frameId , *_src, _params, _bank, true, durations );

    if( durations ) {
        durations->print( std::cerr );
    }

    CCTagFileDebug::instance().outPutAllSessions();
    CCTagFileDebug::instance().clearSessions();
    CCTagVisualDebug::instance().outPutAllSessions();
    CCTagVisualDebug::instance().clearSessions();

    CCTAG_COUT( markers.size() << " markers.");
    CCTAG_COUT("Total time: " << t.elapsed());
    CCTAG_COUT_NOENDL("Id : ");

    int i = 0;
    *_output << "#frame " << _frameId << '\n';
    *_output << _markers.size() << '\n';
    BOOST_FOREACH(const cctag::CCTag & marker, _markers) {
      *_output << marker.x() << " " << marker.y() << " " << marker.id() << " " << marker.getStatus() << '\n';
      if (i == 0) {
          CCTAG_COUT_NOENDL(marker.id() + 1);
      } else {
          CCTAG_COUT_NOENDL(", " << marker.id() + 1);
      }
      ++i;
    }
    CCTAG_COUT("");
}


} //namespace popart
