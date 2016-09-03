/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include "cctag/utils/FileDebug.hpp"
#include "cctag/utils/VisualDebug.hpp"
#include "cctag/utils/Exceptions.hpp"
#include "cctag/Detection.hpp"
#include "CmdLine.hpp"

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
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <opencv/cv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>

#include <tbb/tbb.h>

#define PRINT_TO_CERR

using namespace cctag;
using boost::timer;

using namespace boost::gil;
namespace bfs = boost::filesystem;

void drawMarkers(const boost::ptr_list<CCTag> &markers, cv::Mat &image)
{
  BOOST_FOREACH(const cctag::CCTag & marker, markers)
  {
    if(marker.getStatus() == 1)
    {
      cv::circle(image, cv::Point(marker.x(), marker.y()), 10, cv::Scalar(0, 255, 0 , 255), 3);
      cv::putText(image, std::to_string(marker.id()), cv::Point(marker.x(), marker.y()), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 255, 0, 255), 3);
    }
    else
    {
      cv::circle(image, cv::Point(marker.x(), marker.y()), 10, cv::Scalar(0, 0, 255 , 255), 2);
      cv::putText(image, std::to_string(marker.id()), cv::Point(marker.x(), marker.y()), cv::FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 0, 255, 255), 3);
      
    }

  }
}

void detection(std::size_t frameId,
               int pipeId,
               const cv::Mat & src,
               const cctag::Parameters & params,
               const cctag::CCTagMarkersBank & bank,
               boost::ptr_list<CCTag> &markers,
               std::ostream & output,
               std::string debugFileName = "")
{

  if(debugFileName == "")
  {
    debugFileName = "00000";
  }

  // Process markers detection
  boost::timer t;

  CCTagVisualDebug::instance().initBackgroundImage(src);
  CCTagVisualDebug::instance().setImageFileName(debugFileName);
  CCTagFileDebug::instance().setPath(CCTagVisualDebug::instance().getPath());

  static cctag::logtime::Mgmt* durations = 0;

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

  std::size_t i = 0;
  std::size_t nMarkers = 0;
  output << "#frame " << frameId << '\n';
  output << "Detected " << markers.size() << " candidates" << '\n';

  BOOST_FOREACH(const cctag::CCTag & marker, markers)
  {
    output << marker.x() << " " << marker.y() << " " << marker.id() << " " << marker.getStatus() << '\n';
    ++i;
    if(marker.getStatus() == 1)
      ++nMarkers;
  }
  
  i = 0;
  BOOST_FOREACH(const cctag::CCTag & marker, markers)
  {
    if(i == 0)
    {
      CCTAG_COUT_NOENDL(marker.id() + 1);
    }
    else
    {
      CCTAG_COUT_NOENDL(", " << marker.id() + 1);
    }
    ++i;
  }

  std::cout << std::endl << nMarkers << " markers detected and identified" << std::endl;
}

/*************************************************************/
/*                    Main entry                             */

/*************************************************************/
int main(int argc, char** argv)
{
  CmdLine cmdline;

  if(cmdline.parse(argc, argv) == false)
  {
    cmdline.usage(argv[0]);
    return EXIT_FAILURE;
  }

  cmdline.print(argv[0]);

  // Check input path
  if(!cmdline._filename.empty())
  {
    if(!boost::filesystem::exists(cmdline._filename))
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
  popart::pop_cuda_only_sync_calls(cmdline._switchSync);
#endif

  // Check the (optional) parameters path
  std::size_t nCrowns = std::atoi(cmdline._nCrowns.c_str());
  cctag::Parameters params(nCrowns);

  if(cmdline._paramsFilename != "")
  {
    if(!boost::filesystem::exists(cmdline._paramsFilename))
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
    assert(nCrowns == params._nCrowns);
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

  if(cmdline._debugDir != "")
  {
    params.setDebugDir(cmdline._debugDir);
  }

  popart::device_prop_t deviceInfo(false);
#endif // WITH_CUDA

  bfs::path myPath(cmdline._filename);
  std::string ext(myPath.extension().string());

  const bfs::path parentPath(myPath.parent_path() == "" ? "." : myPath.parent_path());
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

  if((ext == ".png") || (ext == ".jpg") || (ext == ".PNG") || (ext == ".JPG"))
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
#else
    detection(0, pipeId, graySrc, params, bank, markers, outputFile, myPath.stem().string());
#endif
  }
  else if(ext == ".avi" || ext == ".mov")
  {
    CCTAG_COUT("*** Video mode ***");
    POP_INFO("looking at video " << myPath.string());

    // open video and check
    cv::VideoCapture video(cmdline._filename);
    if(!video.isOpened())
    {
      CCTAG_COUT("Unable to open the video : " << cmdline._filename);
      return EXIT_FAILURE;
    }

    // play loop
    int lastFrame = video.get(CV_CAP_PROP_FRAME_COUNT);

    std::list<cv::Mat*> frames;

    std::cerr << "Starting to read video frames" << std::endl;
    while(video.get(CV_CAP_PROP_POS_FRAMES) < lastFrame)
    {
      cv::Mat frame;
      video >> frame;
      cv::Mat* imgGray = new cv::Mat;

      if(frame.channels() == 3 || frame.channels() == 4)
        cv::cvtColor(frame, *imgGray, cv::COLOR_BGR2GRAY);
      else
        frame.copyTo(*imgGray);

      frames.push_back(imgGray);
    }
    std::cerr << "Done. Now processing." << std::endl;

    boost::timer t;
    std::size_t frameId = 0;

    for(cv::Mat* imgGray : frames)
    {
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
      detection(frameId, pipeId, *imgGray, params, bank, markers, std::cerr, outFileName.str());
#else
      detection(frameId, pipeId, *imgGray, params, bank, markers, outputFile, outFileName.str());
#endif
      ++frameId;
      if(frameId % 100 == 0)
      {
        std::cerr << frameId << " (" << std::setprecision(3) << t.elapsed()*1000.0 / frameId << ") ";
      }
    }
    std::cerr << std::endl;
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
    throw std::logic_error("Unrecognized input.");
  }
  outputFile.close();
  return EXIT_SUCCESS;
}

