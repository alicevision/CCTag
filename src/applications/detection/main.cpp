/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cctag/utils/FileDebug.hpp"
#include "cctag/utils/VisualDebug.hpp"
#include "cctag/utils/Exceptions.hpp"
#include "cctag/Detection.hpp"
#include "CmdLine.hpp"

#ifdef WITH_CUDA
#include "cctag/cuda/device_prop.hpp"
#include "cctag/cuda/debug_macros.hpp"
#endif // WITH_CUDA

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
#include <boost/exception/all.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <opencv/cv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

#include <tbb/tbb.h>

#define PRINT_TO_CERR

using namespace cctag;
using boost::timer;

namespace bfs = boost::filesystem;

/**
 * @brief Check if a string is an integer number.
 * 
 * @param[in] s The string to check.
 * @return Return true if the string is an integer number
 */
bool isInteger(std::string &s)
{
  return (s.size() == 1 && std::isdigit(s[0]));
}

/**
 * @brief Draw the detected marker int the given image. The markers are drawn as a
 * circle centered in the center of the marker and with its id. It draws the 
 * well identified markers in green, the unknown / badly detected markers in red.
 * 
 * @param[in] markers The list of markers to draw.
 * @param[out] image The image in which to draw the markers.
 */
void drawMarkers(const boost::ptr_list<CCTag> &markers, cv::Mat &image)
{
  for(const cctag::CCTag & marker : markers)
  {
    const cv::Point center = cv::Point(marker.x(), marker.y());
    const int radius = 10;
    const int fontSize = 3;
    if(marker.getStatus() == status::id_reliable)
    {
      const cv::Scalar color = cv::Scalar(0, 255, 0 , 255);
      cv::circle(image, center, radius, color, 3);
      cv::putText(image, std::to_string(marker.id()), center, cv::FONT_HERSHEY_SIMPLEX, fontSize, color, 3);
    }
    else
    {
      const cv::Scalar color = cv::Scalar(0, 0, 255 , 255);
      cv::circle(image, center, radius, color, 2);
      cv::putText(image, std::to_string(marker.id()), center, cv::FONT_HERSHEY_SIMPLEX, fontSize, color, 3);
    }

  }
}

/**
 * @brief Extract the cctag from an image.
 * 
 * @param[in] frameId The number of the frame.
 * @param[in] pipeId The pipe id (used for multiple streams).
 * @param[in] src The image to process.
 * @param[in] params The parameters for the detection.
 * @param[in] bank The marker bank.
 * @param[out] markers The list of detected markers.
 * @param[out] outStream The output stream on which to write debug information.
 * @param[out] debugFileName The filename for the image to save with the detected 
 * markers.
 */
void detection(std::size_t frameId,
               int pipeId,
               const cv::Mat & src,
               const cctag::Parameters & params,
               const cctag::CCTagMarkersBank & bank,
               boost::ptr_list<CCTag> &markers,
               std::ostream & outStream,
               std::string debugFileName = "")
{

  if(debugFileName.empty())
  {
    debugFileName = "00000";
  }

  // Process markers detection
  boost::timer t;

  CCTagVisualDebug::instance().initBackgroundImage(src);
  CCTagVisualDebug::instance().setImageFileName(debugFileName);
  CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

  static cctag::logtime::Mgmt* durations = nullptr;

  //Call the main CCTag detection function
  cctagDetection(markers, pipeId, frameId, src, params, bank, true, durations);

  if(durations)
  {
    durations->print(std::cerr);
  }

  CCTagFileDebug::instance().outPutAllSessions();
  CCTagFileDebug::instance().clearSessions();
  CCTagVisualDebug::instance().outPutAllSessions();
  CCTagVisualDebug::instance().clearSessions();

  std::cout << "Total time: " << t.elapsed() << std::endl;
  CCTAG_COUT_NOENDL("Id : ");

  std::size_t counter = 0;
  std::size_t nMarkers = 0;
  outStream << "#frame " << frameId << '\n';
  outStream << "Detected " << markers.size() << " candidates" << '\n';

  for(const cctag::CCTag & marker : markers)
  {
    outStream << marker.x() << " " << marker.y() << " " << marker.id() << " " << marker.getStatus() << '\n';
    ++counter;
    if(marker.getStatus() == status::id_reliable)
      ++nMarkers;
  }
  
  counter = 0;
  for(const cctag::CCTag & marker : markers)
  {
    if(counter == 0)
    {
      CCTAG_COUT_NOENDL(marker.id() + 1);
    }
    else
    {
      CCTAG_COUT_NOENDL(", " << marker.id() + 1);
    }
    ++counter;
  }

  std::cout << std::endl << nMarkers << " markers detected and identified" << std::endl;
}

/*************************************************************/
/*                    Main entry                             */

/*************************************************************/
int main(int argc, char** argv)
{
  CmdLine cmdline;

  if(!cmdline.parse(argc, argv))
  {
    cmdline.usage(argv[0]);
    return EXIT_FAILURE;
  }

  cmdline.print(argv[0]);
  
  bool useCamera = false;

  // Check input path
  if(!cmdline._filename.empty())
  {
    if(isInteger(cmdline._filename))
    {
      useCamera = true;
    }
    else if(!bfs::exists(cmdline._filename))
    {
      std::cerr << std::endl
              << "The input file \"" << cmdline._filename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }
  }
  else
  {
    std::cerr << std::endl
            << "An input file is required" << std::endl;
    cmdline.usage(argv[0]);
    return EXIT_FAILURE;
  }

#ifdef WITH_CUDA
  cctag::pop_cuda_only_sync_calls(cmdline._switchSync);
#endif

  // Check the (optional) parameters path
  const std::size_t nCrowns = cmdline._nRings;
  cctag::Parameters params(nCrowns);

  if(!cmdline._paramsFilename.empty())
  {
    if(!bfs::exists(cmdline._paramsFilename))
    {
      std::cerr << std::endl
              << "The input file \"" << cmdline._paramsFilename << "\" is missing" << std::endl;
      return EXIT_FAILURE;
    }

    // Read the parameter file provided by the user
    std::ifstream ifs(cmdline._paramsFilename);
    boost::archive::xml_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT(params._nCrowns);
    CCTAG_COUT(nCrowns);
    if(nCrowns != params._nCrowns)
    {
      std::cerr << std::endl
              << "The number of rings is inconsistent between the parameter file (" 
              << params._nCrowns << ") and the command line (" 
              << nCrowns << ")" << std::endl;
      return EXIT_FAILURE;
    }
  }
  else
  {
    // Use the default parameters and save them in defaultParameters.xml
    cmdline._paramsFilename = "defaultParameters.xml";
    std::ofstream ofs(cmdline._paramsFilename);
    boost::archive::xml_oarchive oa(ofs);
    oa << boost::serialization::make_nvp("CCTagsParams", params);
    CCTAG_COUT("Parameter file not provided. Default parameters are used.");
  }

  CCTagMarkersBank bank(params._nCrowns);
  if(!cmdline._cctagBankFilename.empty())
  {
    bank = CCTagMarkersBank(cmdline._cctagBankFilename);
  }

#ifdef WITH_CUDA
  if(cmdline._useCuda)
  {
    params.setUseCuda(true);
  }
  else
  {
    params.setUseCuda(false);
  }

  if(!cmdline._debugDir.empty())
  {
    params.setDebugDir(cmdline._debugDir);
  }

  cctag::device_prop_t deviceInfo(false);
#endif // WITH_CUDA

  bfs::path myPath(bfs::absolute(cmdline._filename));
  std::string ext(myPath.extension().string());
  boost::algorithm::to_lower(ext);

  const bfs::path parentPath(myPath.parent_path());
  std::string outputFileName;
  if(!bfs::is_directory(myPath))
  {
    CCTagVisualDebug::instance().initializeFolders(parentPath, cmdline._outputFolderName, params._nCrowns);
    outputFileName = parentPath.string() + "/" + cmdline._outputFolderName + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }
  else
  {
    CCTagVisualDebug::instance().initializeFolders(myPath, cmdline._outputFolderName, params._nCrowns);
    outputFileName = myPath.string() + "/" + cmdline._outputFolderName + "/cctag" + std::to_string(nCrowns) + "CC.out";
  }
  std::ofstream outputFile;
  outputFile.open(outputFileName);

#if USE_DEVIL
  if( (ext == ".bmp") ||
      (ext == ".gif") ||
      (ext == ".jpg") ||
      (ext == ".lbm") ||
      (ext == ".pbm") ||
      (ext == ".pgm") ||
      (ext == ".png") ||
      (ext == ".ppm") ||
      (ext == ".tga") ||
      (ext == ".tif") )
  {
    std::cout << "******************* Image mode **********************" << std::endl;
    POP_INFO("looking at image " << myPath.string());
    ilImage img;
    if( img.Load( cmdline._filename.c_str() ) == false )
    {
      std::cerr << "Could not load image " << cmdline._filename << std::endl;
      return 0;
    }
    if( img.Convert( IL_LUMINANCE ) == false )
    {
      std::cerr << "Failed converting image " << cmdline._filename << " to unsigned greyscale image" << std::endl;
      exit( -1 );
    }
    int w = img.Width();
    int h = img.Height();
    std::cout << "Loading " << w << " x " << h << " image " << cmdline._filename << std::endl;
    unsigned char* image_data = img.GetData();
    cv::Mat graySrc = cv::Mat( h, w, CV_8U, image_data );

    imwrite( "ballo.jpg", graySrc );

    const int pipeId = 0;
    boost::ptr_list<CCTag> markers;
#ifdef PRINT_TO_CERR
    detection(0, pipeId, graySrc, params, bank, markers, std::cerr, myPath.stem().string());
#else // PRINT_TO_CERR
    detection(0, pipeId, graySrc, params, bank, markers, outputFile, myPath.stem().string());
#endif // PRINT_TO_CERR
  }
#else // USE_DEVIL
  if((ext == ".png") || (ext == ".jpg"))
  {

    std::cout << "******************* Image mode **********************" << std::endl;

    POP_INFO("looking at image " << myPath.string());

    // Gray scale conversion
    cv::Mat src = cv::imread(cmdline._filename);
    cv::Mat graySrc;
    cv::cvtColor(src, graySrc, CV_BGR2GRAY);

    const int pipeId = 0;
    boost::ptr_list<CCTag> markers;
#ifdef PRINT_TO_CERR
    detection(0, pipeId, graySrc, params, bank, markers, std::cerr, myPath.stem().string());
#else // PRINT_TO_CERR
    detection(0, pipeId, graySrc, params, bank, markers, outputFile, myPath.stem().string());
#endif // PRINT_TO_CERR
  }
#endif // USE_DEVIL
  else if(ext == ".avi" || ext == ".mov" || useCamera)
  {
    CCTAG_COUT("*** Video mode ***");
    POP_INFO("looking at video " << myPath.string());

    // open video and check
    cv::VideoCapture video;
    if(useCamera)
      video.open(std::atoi(cmdline._filename.c_str()));
    else
      video.open(cmdline._filename);
    
    if(!video.isOpened())
    {
      CCTAG_COUT("Unable to open the video : " << cmdline._filename);
      return EXIT_FAILURE;
    }

    const std::string windowName = "Detection result";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    std::cerr << "Starting to read video frames" << std::endl;
    std::size_t frameId = 0;
    
    // time to wait in milliseconds for keyboard input, used to switch from
    // live to debug mode
    int delay = 10;

    while(true)
    {
      cv::Mat frame;
      video >> frame;
      if(frame.empty())
        break;
      
      cv::Mat imgGray;

      if(frame.channels() == 3 || frame.channels() == 4)
        cv::cvtColor(frame, imgGray, cv::COLOR_BGR2GRAY);
      else
        frame.copyTo(imgGray);

      // Set the output folder
      std::stringstream outFileName;
      outFileName << std::setfill('0') << std::setw(5) << frameId;
      
      boost::ptr_list<CCTag> markers;

      // Invert the image for the projection scenario
      //cv::Mat imgGrayInverted;
      //bitwise_not ( imgGray, imgGrayInverted );

      // Call the CCTag detection
      const int pipeId = 0;
#ifdef PRINT_TO_CERR
      detection(frameId, pipeId, imgGray, params, bank, markers, std::cerr, outFileName.str());
#else
      detection(frameId, pipeId, imgGray, params, bank, markers, outputFile, outFileName.str());
#endif
      
      // if the original image is b/w convert it to BGRA so we can draw colors
      if(frame.channels() == 1)
        cv::cvtColor(imgGray, frame, cv::COLOR_GRAY2BGRA);
      
      drawMarkers(markers, frame);
      cv::imshow(windowName, frame);
      if( cv::waitKey(delay) == 27 ) break;
      char key = (char) cv::waitKey(delay);
      // stop capturing by pressing ESC
      if(key == 27) 
        break;
      if(key == 'l' || key == 'L')
        delay = 10;
      // delay = 0 will wait for a key to be pressed
      if(key == 'd' || key == 'D')
        delay = 0;
      
      ++frameId;
    }

  }
  else if(bfs::is_directory(myPath))
  {
    CCTAG_COUT("*** Image sequence mode ***");

    std::vector<bfs::path> vFileInFolder;

    std::copy(bfs::directory_iterator(myPath), bfs::directory_iterator(), std::back_inserter(vFileInFolder)); // is directory_entry, which is
    std::sort(vFileInFolder.begin(), vFileInFolder.end());

    std::size_t frameId = 0;

    std::map<int, bfs::path> files[2];
    for(const auto & fileInFolder : vFileInFolder)
    {
      files[frameId & 1].insert(std::pair<int, bfs::path>(frameId, fileInFolder));
      frameId++;
    }

    tbb::parallel_for(0, 2, [&](size_t fileListIdx)
    {
      for(const auto & fileInFolder : files[fileListIdx])
      {
        const std::string subExt(bfs::extension(fileInFolder.second));

        if((subExt == ".png") || (subExt == ".jpg") || (subExt == ".PNG") || (subExt == ".JPG"))
        {

          std::cerr << "Processing image " << fileInFolder.second.string() << std::endl;

          cv::Mat src;
          src = cv::imread(fileInFolder.second.string());

          cv::Mat imgGray;
          cv::cvtColor(src, imgGray, CV_BGR2GRAY);

          // Call the CCTag detection
          int pipeId = (fileInFolder.first & 1);
          boost::ptr_list<CCTag> markers;
#ifdef PRINT_TO_CERR
          detection(fileInFolder.first, pipeId, imgGray, params, bank, markers, std::cerr, fileInFolder.second.stem().string());
#else
          detection(fileInFolder.first, pipeId, imgGray, params, bank, markers, outputFile, fileInFolder.second.stem().string());
#endif
          std::cerr << "Done processing image " << fileInFolder.second.string() << std::endl;
        }
      }
    });
  }
  else
  {
    std::cerr << "The input file format is not supported" << std::endl;
    throw std::logic_error("Unrecognized input.");
  }
  outputFile.close();
  return EXIT_SUCCESS;
}

