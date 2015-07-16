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

void detection(std::size_t frame, cctag::View& view, const cctag::Parameters & params, const cctag::CCTagMarkersBank & bank, std::ofstream & output, std::string debugFileName = "")
{
    POP_ENTER;
    
    if (debugFileName == "") {
      debugFileName = "00000";
    }
    
    // Process markers detection
    boost::timer t;
    boost::ptr_list<CCTag> markers;

    view.setNumLayers( params._numberOfMultiresLayers );
    
    CCTagVisualDebug::instance().initBackgroundImage(view._view);
    CCTagVisualDebug::instance().setImageFileName(debugFileName);
    CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());
    
    cctagDetection(markers, frame, view._grayView, params, bank, true );
    
    CCTagFileDebug::instance().outPutAllSessions();
    CCTagFileDebug::instance().clearSessions();
    CCTagVisualDebug::instance().outPutAllSessions();
    CCTagVisualDebug::instance().clearSessions();

    CCTAG_COUT( markers.size() << " markers.");
    CCTAG_COUT("Total time: " << t.elapsed());
    CCTAG_COUT_NOENDL("Id : ");

    int i = 0;
    output << "#frame " << frame << '\n';
    output << markers.size() << '\n';
    BOOST_FOREACH(const cctag::CCTag & marker, markers) {
      output << marker.x() << " " << marker.y() << " " << marker.id() << " " << marker.getStatus() << '\n';
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
    boost::archive::xml_iarchive ia(ifs);
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
    CCTagVisualDebug::instance().initializeFolders( parentPath , params._nCrowns );
    outputFileName = parentPath.string() + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }else
  {
    CCTagVisualDebug::instance().initializeFolders( myPath , params._nCrowns );
    outputFileName = myPath.string() + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }
  std::ofstream outputFile;
  outputFile.open( outputFileName );
  
  if ( (ext == ".png") || (ext == ".jpg") ) {

    POP_INFO( "looking at image " << myPath.string() );

    cctag::View my_view( cmdline._filename );

    // Increase image size.
    /*{
            rgb8_image_t simage;
            simage.recreate( 2 * image.width(), 2 * image.height() );
            rgb8_view_t osvw( view( simage ) );
            terry::resize_view( svw, osvw, terry::sampler::bicubic_sampler() );
    }*/

    // Call the CCTag detection
    detection(0, my_view, params, bank, outputFile, myPath.stem().string());
    
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
    std::size_t frameId = 0;
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
      detection(frameId, cctagView, params, bank, outputFile, outFileName.str());
      
      ++frameId; 
    }
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

        cctag::View my_view( fileInFolder.string() );
        
        // Call the CCTag detection
        detection(frameId, my_view, params, bank, outputFile, fileInFolder.stem().string());
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

