#include <cctag/ICCTag.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/detection.hpp>

#include <boost/foreach.hpp>

namespace cctag {

void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const boost::gil::gray8_view_t& graySrc,
      const cctag::vision::marker::Parameters & params,
      const bool bDisplayEllipses)
{
  
  markers.clear();// need to be checked!
  boost::ptr_list<cctag::vision::marker::CCTag> cctags;
  
  cctag::vision::marker::cctagDetection(cctags,
        frame, 
        graySrc,
        params, bDisplayEllipses);
  
  
  BOOST_FOREACH(const cctag::vision::marker::CCTag & cctag, cctags)
  {
    markers.push_back(new cctag::vision::marker::CCTag(cctag));
  }
}

}

