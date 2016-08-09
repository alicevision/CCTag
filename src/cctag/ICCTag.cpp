#include <cctag/ICCTag.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/Detection.hpp>
#include <cctag/utils/LogTime.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <fstream>

using namespace std;

namespace cctag {

/**
 * @brief Perform the CCTag detection on a gray scale image
 * 
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are valid ones. (status available via getStatus()) 
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] graySrc Gray scale input image.
 * @param[in] nRings Number of CCTag rings.
 * @param[in] parameterFile Path to a parameter file. If not provided default parameters will be used.
 * @param[in] cctagBankFilename Path to the cctag bank. If not provided, radii will be the ones associated to the CCTags contained in the
 * markersToPrint folder.
 */
void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const cv::Mat & graySrc,
      const std::size_t nRings,
      logtime::Mgmt* durations,
      const std::string & parameterFilename,
      const std::string & cctagBankFilename)
{
  // Load parameters
  cctag::Parameters params = cctag::Parameters(nRings);
    
  if ( !parameterFilename.empty() )
  {
    if (!boost::filesystem::exists( parameterFilename )) {
      std::cerr << std::endl
        << "The input parameter file \""<< parameterFilename << "\" is missing" << std::endl;
      return;
    }else{
      std::ifstream ifs( parameterFilename.c_str() );
      boost::archive::xml_iarchive ia(ifs);
      ia >> boost::serialization::make_nvp("CCTagsParams", params);
      assert(  nRings == params._nCrowns  );
    }
  }
  
  CCTagMarkersBank bank(params._nCrowns);
  if ( !cctagBankFilename.empty())
  {
    bank = CCTagMarkersBank(cctagBankFilename);
  }
  
  cctagDetection(markers, frame, graySrc, params, durations, &bank);
}

void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const cv::Mat & graySrc,
      const cctag::Parameters & params,
      logtime::Mgmt* durations,
      const CCTagMarkersBank * pBank)
{
  boost::ptr_list<cctag::CCTag> cctags;
  
  if ( pBank == NULL)
  {
    CCTagMarkersBank bank(params._nCrowns);
    cctag::cctagDetection(cctags, frame, graySrc, params, bank, false, durations);
  }else
  {
    cctag::cctagDetection(cctags, frame, graySrc, params, *pBank, false, durations);
  }
  
  markers.clear();
  BOOST_FOREACH(const cctag::CCTag & cctag, cctags)
  {
    markers.push_back(new cctag::CCTag(cctag));
  }
}


}

