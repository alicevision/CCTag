#include <cctag/ICCTag.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/detection.hpp>

#include <boost/foreach.hpp>

namespace cctag {

void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const boost::gil::gray8_view_t& graySrc,
      const cctag::Parameters & params,
      const bool bDisplayEllipses)
{
  std::string cctagBankFilename("/home/lilian/cpp_workspace/CCTag/cctagLibraries/3Crowns/ids.txt");
  
  markers.clear();// need to be checked!
  boost::ptr_list<cctag::CCTag> cctags;
  
  cctag::cctagDetection(cctags,
        frame, 
        graySrc,
        params, 
        cctagBankFilename,
        bDisplayEllipses);
  
  
  BOOST_FOREACH(const cctag::CCTag & cctag, cctags)
  {
    markers.push_back(new cctag::CCTag(cctag));
  }
}

}

