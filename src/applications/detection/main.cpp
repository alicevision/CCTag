#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/cmdline.hpp"

#ifdef WITH_CUDA
#include "cuda/device_prop.hpp"
#include "cuda/debug_macros.hpp"
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

#include <opencv/cv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs   = boost::filesystem;
namespace btime = boost::posix_time;

// static const std::string kUsageString = "Usage: detection image_file.png\n";

void detection(std::size_t frameId, const cv::Mat & src, const cctag::Parameters & params, const cctag::CCTagMarkersBank & bank, std::ofstream & output, std::string debugFileName = "")
{
    POP_ENTER;
    
    if (debugFileName == "") {
      debugFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;
    
    CCTagVisualDebug::instance().initBackgroundImage(src);
    CCTagVisualDebug::instance().setImageFileName(debugFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

    static cctag::logtime::Mgmt* durations;
std::cerr << "before ccTagDetection" << std::endl;
    if( not durations ) {
        durations = new cctag::logtime::Mgmt( 25 );
    } else {
        durations->resetStartTime();
    }
    cctagDetection( markers, frameId , src, params, bank, true, durations );
    durations->print( std::cerr );
std::cerr << "after ccTagDetection" << std::endl;

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
    POP_LEAVE;
}

/*************************************************************/
/*                    Main entry                             */
/*************************************************************/
int main(int argc, char** argv)
{
  if( cmdline.parse( argc, argv ) == false ) {
    cmdline.usage( argv[0] );
    return EXIT_FAILURE;
  }

  cmdline.print( argv[0] );

  // Check input path
  if( cmdline._filename.compare("") != 0){
    if (!boost::filesystem::exists( cmdline._filename )) {
      std::cerr << std::endl
        << "The input file \""<< cmdline._filename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }else{
    std::cerr << std::endl
        << "An input file is required" << std::endl;
    cmdline.usage( argv[0] );
    return EXIT_FAILURE;
  }

#ifdef WITH_CUDA
  popart::pop_cuda_only_sync_calls( cmdline._switchSync );
#endif

  // Check the (optional) parameters path
  std::size_t nCrowns = std::atoi(cmdline._nCrowns.c_str());
  cctag::Parameters params(nCrowns);
  
  if( cmdline._paramsFilename != "" ) {
    if (!boost::filesystem::exists( cmdline._paramsFilename )) {
      std::cerr << std::endl
        << "The input file \""<< cmdline._paramsFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }

    // Read the parameter file provided by the user
    std::ifstream ifs( cmdline._paramsFilename.c_str() );
    boost::archive::xml_iarchive ia( ifs );
    ia >> boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT(params._nCrowns);
    CCTAG_COUT(nCrowns);
    assert( nCrowns == params._nCrowns );
  } else {
    // Use the default parameters and save them in defaultParameters.xml
    cmdline._paramsFilename = "defaultParameters.xml";
    std::ofstream ofs( cmdline._paramsFilename.c_str() );
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT("Parameter file not provided. Default parameters are used.");
  }
  
  CCTagMarkersBank bank(params._nCrowns);
  if ( !cmdline._cctagBankFilename.empty())
  {
    bank = CCTagMarkersBank(cmdline._cctagBankFilename);
  }

  if( cmdline._debugDir != "" ) {
    params.setDebugDir( cmdline._debugDir );
  }

#ifdef WITH_CUDA
  popart::device_prop_t deviceInfo;
  deviceInfo.print( );
#endif // WITH_CUDA

  bfs::path myPath( cmdline._filename );
  std::string ext(myPath.extension().string());

  const bfs::path subFilenamePath(myPath.filename());
  const bfs::path parentPath( myPath.parent_path() == "" ? "." : myPath.parent_path());
  std::string outputFileName;
  if (!bfs::is_directory(myPath))
  {
    CCTagVisualDebug::instance().initializeFolders( parentPath , cmdline._outputFolderName , params._nCrowns );
    outputFileName = parentPath.string() + "/" + cmdline._outputFolderName + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }else
  {
    CCTagVisualDebug::instance().initializeFolders( myPath , cmdline._outputFolderName , params._nCrowns );
    outputFileName = myPath.string() + "/" + cmdline._outputFolderName + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }
  std::ofstream outputFile;
  outputFile.open( outputFileName );
  
  if ( (ext == ".png") || (ext == ".jpg") || (ext == ".PNG") || (ext == ".JPG")) {
    POP_INFO("looking at image " << myPath.string());
    
    // Gray scale convertion
    cv::Mat src = cv::imread(cmdline._filename);
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
    detection(0, graySrc, params, bank, outputFile, myPath.stem().string());
  //detection(0, my_view, params, bank, outputFile, myPath.stem().string());
} else if (ext == ".avi" )
  {
    CCTAG_COUT("*** Video mode ***");
    POP_INFO( "looking at video " << myPath.string() );

    // open video and check
    cv::VideoCapture video( cmdline._filename.c_str() );
    if(!video.isOpened())
    {
      CCTAG_COUT("Unable to open the video : " << cmdline._filename); return -1;
    }

    // play loop
    int lastFrame = video.get(CV_CAP_PROP_FRAME_COUNT);

    std::list<cv::Mat*> frames;

    std::cerr << "Starting to read video frames" << std::endl;
    while( video.get(CV_CAP_PROP_POS_FRAMES) < lastFrame )
    {
      cv::Mat frame;
      video >> frame;
      cv::Mat* imgGray = new cv::Mat;
      cv::cvtColor( frame, *imgGray, CV_BGR2GRAY );

      frames.push_back( imgGray );
    }
    std::cerr << "Done. Now processing." << std::endl;

    boost::timer t;
    std::size_t frameId = 0;
    for( cv::Mat* imgGray : frames ) {
        // Set the output folder
        std::stringstream outFileName;
        outFileName << std::setfill('0') << std::setw(5) << frameId;

        // Invert the image for the projection scenario
        //cv::Mat imgGrayInverted;
        //bitwise_not ( imgGray, imgGrayInverted );
      
        // Call the CCTag detection
        detection(frameId, *imgGray, params, bank, outputFile, outFileName.str());
        ++frameId; 
        if( frameId % 100 == 0 ) {
            std::cerr << frameId << " (" << std::setprecision(3) << t.elapsed()*1000.0/frameId << ") ";
        }
    }
    std::cerr << std::endl;
  } else if (bfs::is_directory(myPath)) {
    CCTAG_COUT("*** Image sequence mode ***");

    std::vector<bfs::path> vFileInFolder;

    std::copy(bfs::directory_iterator(myPath), bfs::directory_iterator(), std::back_inserter(vFileInFolder)); // is directory_entry, which is
    std::sort(vFileInFolder.begin(), vFileInFolder.end());

    std::size_t frameId = 0;

    for(const auto & fileInFolder : vFileInFolder) {
      const std::string subExt(bfs::extension(fileInFolder));
      
      if ( (subExt == ".png") || (subExt == ".jpg") ) {
        CCTAG_COUT( "Processing image " << fileInFolder.string() );

		cv::Mat src;
    	src = cv::imread(fileInFolder.string());

        cv::Mat imgGray;
        cv::cvtColor( src, imgGray, CV_BGR2GRAY );
      
        // Call the CCTag detection
        detection(frameId, imgGray, params, bank, outputFile, fileInFolder.stem().string());
++frameId;
      }
    }
  }else
  {
      throw std::logic_error("Unrecognized input.");
  }
  outputFile.close();
  return 0;
}

