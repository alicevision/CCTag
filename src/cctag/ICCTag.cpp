#include <cctag/ICCTag.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/detection.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <fstream>

namespace cctag {

void cctagDetection(
      boost::ptr_list<ICCTag> & markers,
      const std::size_t frame,
      const boost::gil::gray8_view_t& graySrc,
      const std::size_t nCrowns,
      const std::string & parameterFilename,
      const std::string & cctagBankFilename)
{
  // Load parameters
  cctag::Parameters params = cctag::Parameters(nCrowns);
    
  if (!boost::filesystem::exists( parameterFilename )) {
    std::cerr << std::endl
      << "The input parameter file \""<< parameterFilename << "\" is missing" << std::endl;
    return;
  }else{
    std::ifstream ifs( parameterFilename.c_str() );
    boost::archive::xml_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("CCTagsParams", params);
    assert(  nCrowns == params._nCrowns  );
  }
  
  CCTagMarkersBank bank(params._nCrowns);
  if ( !cctagBankFilename.empty())
  {
    bank = CCTagMarkersBank(cctagBankFilename);
  }
  
  boost::ptr_list<cctag::CCTag> cctags;
  
  cctag::cctagDetection(cctags, frame, graySrc, params, bank);
  
  markers.clear();
  BOOST_FOREACH(const cctag::CCTag & cctag, cctags)
  {
    markers.push_back(new cctag::CCTag(cctag));
  }
}

}

