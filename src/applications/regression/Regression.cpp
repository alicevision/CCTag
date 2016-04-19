#include "Regression.h"

void detection(std::size_t frameId, const cv::Mat & src, const cctag::Parameters & params, const cctag::CCTagMarkersBank & bank, std::ostream & output, std::string debugFileName = "")
{
    if (debugFileName == "") {
      debugFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;
    
    CCTagVisualDebug::instance().initBackgroundImage(src);
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
    cctagDetection( markers, frameId , src, params, bank, true, durations );

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
    output << "#frame " << frameId << '\n';
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
