#include "cctag/package.h"

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/cmdline.hpp"

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

namespace popart
{

Package::Package( const cctag::Parameters & params,
                  const cctag::CCTagMarkersBank & bank )
    : _src( 0 )
    , _params( params )
    , _bank( bank )
{
    winners = new WinnerMap[ params._numberOfProcessedMultiresLayers ];
    seeds   = new std::vector<EdgePoint*>[ params._numberOfProcessedMultiresLayers ];
}

void Package::resetTables( )
{
    for( int i=0; i<_params._numberOfProcessedMultiresLayers; i++ ) {
        winners[i].clear();
        seeds  [i].clear();
    }
}

void Package::init( int frameId,
               const cv::Mat* src )
{
    if( _src == 0 ) {
        // allocate memories
    }

    _frameId = frameId;
    _src     = src;
}

void Package::detection( std::ostream & output,
                         std::string debugFileName )
{
    if (debugFileName == "") {
      debugFileName = "00000";
    }

    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;

    CCTagVisualDebug::instance().initBackgroundImage( *_src );
    CCTagVisualDebug::instance().setImageFileName(debugFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

    static cctag::logtime::Mgmt* durations = 0;
#if 0
    if( not durations ) {
        durations = new cctag::logtime::Mgmt( 25 );
    } else {
        durations->resetStartTime();
    }
#endif
    cctagDetection( this, markers, _frameId , *_src, _params, _bank, true, durations );

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
    output << "#frame " << _frameId << '\n';
    output << markers.size() << '\n';
    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
      output << marker.x() << " " << marker.y() << " " << marker.id() << " " << marker.getStatus() << '\n';
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
