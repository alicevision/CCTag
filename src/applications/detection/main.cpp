#include <cctag/fileDebug.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/progBase/exceptions.hpp>
#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/detection.hpp>
#include <cctag/view.hpp>
#include <cctag/image.hpp>

#ifdef WITH_CUDA
#include "cuda/device_prop.hpp"
#endif // WITH_CUDA

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/resample_subimage.hpp>

#include <third_party/cmdLine/cmdLine.h>

#include <opencv/cv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

static const std::string kUsageString = "Usage: detection image_file.png\n";

void detection(std::size_t frameId, const cv::Mat & src, const cctag::Parameters & params, const std::string & cctagBankFilename, std::string outputFileName = "")
{
    POP_ENTER;
    
    if (outputFileName == "") {
      outputFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;
    
    CCTagVisualDebug::instance().initBackgroundImage(src);
    CCTagVisualDebug::instance().setImageFileName(outputFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

    cctagDetection(markers, frameId , src, params, cctagBankFilename, true);
    
    CCTagFileDebug::instance().outPutAllSessions();
    CCTagFileDebug::instance().clearSessions();
    CCTagVisualDebug::instance().outPutAllSessions();
    CCTagVisualDebug::instance().clearSessions();

    CCTAG_COUT( markers.size() << " markers.");
    CCTAG_COUT("Total time: " << t.elapsed());
    CCTAG_COUT_NOENDL("Id : ");

    int i = 0;
    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
        if (i == 0) {
            CCTAG_COUT_NOENDL(marker.id() + 1);
        } else {
            CCTAG_COUT_NOENDL(", " << marker.id() + 1);
        }
        ++i;
    }
    CCTAG_COUT("");
    POP_LEAVE;
}

void printUsageErr( const char* const argv0 )
{
  std::cerr << "Usage: " << argv0 << '\n'
  << "[-i|--input path] \n"
  << "[-b|--bank path] \n"
  << "\n[Optional]\n"
  << "[-p|--params path] \n"
  << std::endl;
}

/*************************************************************/
/*                    Main entry                             */
/*************************************************************/
int main(int argc, char** argv)
{
  cctag::MemoryPool::instance().updateMemoryAuthorizedWithRAM();

  std::string filename = "";
  std::string  cctagBankFilename = "";
  std::string paramsFilename = "";

  CmdLine cmd;
  cmd.add( make_option('i', filename, "input") );
  cmd.add( make_option('b', cctagBankFilename, "cctagBank") );
  cmd.add( make_option('p', paramsFilename, "parameters") );

  try {
      if (argc == 1) throw std::string("Invalid command line parameters.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
    printUsageErr(argv[0]);
    return EXIT_FAILURE;
  }

  std::cout << "You called: " <<std::endl
      << argv[0] << std::endl
      << "--input " << filename << std::endl
      << "--bank " << cctagBankFilename << std::endl
      << "--params " << paramsFilename << std::endl;

  // Check input path
  if (filename.compare("") != 0){
    if (!boost::filesystem::exists(filename)) {
      std::cerr << std::endl
        << "The input file \""<< filename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }else{
    std::cerr << std::endl
        << "An input file is required" << std::endl;
    printUsageErr(argv[0]);
    return EXIT_FAILURE;
  }

  // Check cctag bank path
  if (cctagBankFilename.compare("") != 0){
    if (!boost::filesystem::exists(cctagBankFilename)) {
      std::cerr << std::endl
        << "The input file \""<< cctagBankFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }else{
    std::cerr << std::endl
        << "An bank file is required" << std::endl;
    printUsageErr(argv[0]);
    return EXIT_FAILURE;
  }

  // Check the (optional) parameters path
  cctag::Parameters params;
  
  if (paramsFilename != "") {
    if (!boost::filesystem::exists(paramsFilename)) {
      std::cerr << std::endl
        << "The input file \""<< paramsFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }

    // Read the parameter file provided by the user
    std::ifstream ifs(paramsFilename.c_str());
    boost::archive::xml_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("CCTagsParams", params);
  } else {
    // Use the default parameters and save them in defaultParameters.xml
    paramsFilename = "defaultParameters.xml";
    std::ofstream ofs(paramsFilename.c_str());
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT("Parameter file not provided. Default parameters are used.");
  }

#ifdef WITH_CUDA
  popart::device_prop_t deviceInfo;
  deviceInfo.print( );
#endif // WITH_CUDA

  CCTagVisualDebug::instance().initializeFolders(filename, params._numCrowns);
  bfs::path myPath(filename);
  std::string ext(myPath.extension().string());

  if ( (ext == ".png") || (ext == ".jpg") ) {

    POP_INFO << "looking at image " << myPath.string() << std::endl;
    
    // Gray scale convertion
    cv::Mat src = cv::imread(filename);
    cv::Mat graySrc;
    cv::cvtColor( src, graySrc, CV_BGR2GRAY );

    // Upscale original image
    /*{
            rgb8_image_t simage;
            simage.recreate( 2 * image.width(), 2 * image.height() );
            rgb8_view_t osvw( view( simage ) );
            terry::resize_view( svw, osvw, terry::sampler::bicubic_sampler() );
    }*/

    // Call the CCTag detection
    detection(0, graySrc, params, cctagBankFilename, myPath.stem().string());

  } else if (ext == ".avi" )
  {
    CCTAG_COUT("*** Video mode ***");
    POP_INFO << "looking at video " << myPath.string() << std::endl;

    // open video and check
    cv::VideoCapture video(filename.c_str());
    if(!video.isOpened())
    {
      CCTAG_COUT("Unable to open the video : " << filename); return -1;
    }

    // play loop
    int lastFrame = video.get(CV_CAP_PROP_FRAME_COUNT);
    int frameId = 0;
    while( video.get(CV_CAP_PROP_POS_FRAMES) < lastFrame )
    {
      cv::Mat frame;
      video >> frame;
      cv::Mat imgGray;
      cv::cvtColor( frame, imgGray, CV_BGR2GRAY );

      // Set the output folder
      std::stringstream outFileName;
      outFileName << std::setfill('0') << std::setw(5) << frameId;

      // Call the CCTag detection
      detection(frameId, imgGray, params, cctagBankFilename, outFileName.str());

      ++frameId; 
    }
  } else {
      throw std::logic_error("Unrecognized input.");
  }
  return 0;
}

