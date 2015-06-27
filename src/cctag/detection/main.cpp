#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/progBase/MemoryPool.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/cmdline.hpp"

#if 0
#include <third_party/cmdLine/cmdLine.h>
#endif

#ifdef WITH_CUDA
#include "cuda/device_prop.hpp"
#include "cuda/debug_macros.hpp"
#endif // WITH_CUDA

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/resample_subimage.hpp>

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

// static const std::string kUsageString = "Usage: detection image_file.png\n";

void detection(std::size_t frame, cctag::View& view, const cctag::Parameters & params, const std::string & cctagBankFilename, std::string outputFileName = "")
{
    POP_ENTER;
    
    if (outputFileName == "") {
      outputFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;

    view.setNumLayers( params._numberOfMultiresLayers );
    
    CCTagVisualDebug::instance().initBackgroundImage(view._view);
    CCTagVisualDebug::instance().setImageFileName(outputFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());
    
    cctagDetection(markers, frame, view._grayView, params, cctagBankFilename, true );
    
    CCTagFileDebug::instance().outPutAllSessions();
    CCTagFileDebug::instance().clearSessions();
    CCTagVisualDebug::instance().outPutAllSessions();
    CCTagVisualDebug::instance().clearSessions();

    CCTAG_COUT( markers.size() << " markers.");
    CCTAG_COUT("Total time: " << t.elapsed());
    CCTAG_COUT_NOENDL("Id : ");

    int i = 0;
    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
        cctag::drawMarkerOnGilImage(view._view, marker, false);
        cctag::drawMarkerInfos(view._view, marker, false);

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
  cctag::MemoryPool::instance().updateMemoryAuthorizedWithRAM();
#if 1
  if( cmdline.parse( argc, argv ) == false ) {
    cmdline.usage( argv[0] );
    return EXIT_FAILURE;
  }
#else
  CmdLine cmd;
  cmd.add( OptionField<std::string>('i', filename, "input") );
  // cmd.add( make_option('b', cctagBankFilename, "cctagBank") );
  cmd.add( OptionField<std::string>('b', cctagBankFilename, "bank") );
  cmd.add( OptionField<std::string>('p', paramsFilename, "parameters") );
#ifdef WITH_CUDA
  cmd.add( OptionSwitch( 0xd0, switchSync, "sync" ) );
#endif

  try {
      if (argc == 1) throw std::string("Invalid command line parameters.");
      cmd.process(argc, argv);
  } catch(const std::string& s) {
    printUsageErr(argv[0]);
    return EXIT_FAILURE;
  }
#endif

  cmdline.print( argv[0] );

  // Check input path
  if( cmdline.filename.compare("") != 0){
    if (!boost::filesystem::exists( cmdline.filename )) {
      std::cerr << std::endl
        << "The input file \""<< cmdline.filename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }else{
    std::cerr << std::endl
        << "An input file is required" << std::endl;
    cmdline.usage( argv[0] );
    return EXIT_FAILURE;
  }

  // Check cctag bank path
  if( cmdline.cctagBankFilename.compare("") != 0){
    if (!boost::filesystem::exists( cmdline.cctagBankFilename )) {
      std::cerr << std::endl
        << "The input file \""<< cmdline.cctagBankFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }else{
    std::cerr << std::endl
        << "An bank file is required" << std::endl;
    cmdline.usage(argv[0]);
    return EXIT_FAILURE;
  }

#ifdef WITH_CUDA
  popart::pop_cuda_only_sync_calls( cmdline.switchSync );
#endif
  // Check the (optional) parameters path
  cctag::Parameters params;
  
  if( cmdline.paramsFilename != "" ) {
    if (!boost::filesystem::exists( cmdline.paramsFilename )) {
      std::cerr << std::endl
        << "The input file \""<< cmdline.paramsFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }

    // Read the parameter file provided by the user
    std::ifstream ifs( cmdline.paramsFilename.c_str() );
    boost::archive::xml_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("CCTagsParams", params);
  } else {
    // Use the default parameters and save them in defaultParameters.xml
    cmdline.paramsFilename = "defaultParameters.xml";
    std::ofstream ofs( cmdline.paramsFilename.c_str() );
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT("Parameter file not provided. Default parameters are used.");
  }

#ifdef WITH_CUDA
  popart::device_prop_t deviceInfo;
  deviceInfo.print( );
#endif // WITH_CUDA

  CCTagVisualDebug::instance().initializeFolders( cmdline.filename, params._numCrowns );
  bfs::path myPath( cmdline.filename );
  std::string ext(myPath.extension().string());

  if ( (ext == ".png") || (ext == ".jpg") ) {

    POP_INFO( "looking at image " << myPath.string() );

    cctag::View my_view( cmdline.filename );

    rgb8_image_t& image = my_view._image;
    rgb8_view_t&  svw   = my_view._view;

    // Increase image size.
    /*{
            rgb8_image_t simage;
            simage.recreate( 2 * image.width(), 2 * image.height() );
            rgb8_view_t osvw( view( simage ) );
            terry::resize_view( svw, osvw, terry::sampler::bicubic_sampler() );
    }*/

    // Call the CCTag detection
    detection(0, my_view, params, cmdline.cctagBankFilename, myPath.stem().string());

  } else if (ext == ".avi" )
  {
    CCTAG_COUT("*** Video mode ***");
    POP_INFO( "looking at video " << myPath.string() );

    // open video and check
    cv::VideoCapture video( cmdline.filename.c_str() );
    if(!video.isOpened())
    {
      CCTAG_COUT("Unable to open the video : " << cmdline.filename); return -1;
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
      cctag::View cctagView((const unsigned char *) imgGray.data, imgGray.cols, imgGray.rows , imgGray.step );
      cctagView._view = boost::gil::interleaved_view(imgGray.cols, imgGray.rows, (boost::gil::rgb8_pixel_t*) frame.data, frame.step );

      // Set the output folder
      std::stringstream outFileName;
      outFileName << std::setfill('0') << std::setw(5) << frameId;

      // Call the CCTag detection
      detection(frameId, cctagView, params, cmdline.cctagBankFilename, outFileName.str());

      ++frameId; 
    }
  } else {
      throw std::logic_error("Unrecognized input.");
  }
  return 0;
}

