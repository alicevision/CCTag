/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "TestLog.h"

using namespace cctag;

FrameLog FrameLog::detect(size_t frame, const cv::Mat& src, const Parameters& parameters,
  const cctag::CCTagMarkersBank& bank)
{
  using namespace std::chrono;
  CCTag::List markers;
  
  const auto t0 = high_resolution_clock::now();
  cctagDetection(markers, 0, frame, src, parameters, bank, true, nullptr);
  const auto t1 = high_resolution_clock::now();
  const auto td = duration_cast<milliseconds>(t1 - t0).count() / 1000.f;
  
  return FrameLog(frame, td, markers);
}

/////////////////////////////////////////////////////////////////////////////

static const char* TOPLEVEL_XML_ELEMENT("FileLog");

void FileLog::save(const std::string& filename)
{
  std::ofstream ofs(filename);
  boost::archive::xml_oarchive oa(ofs);
  oa << boost::serialization::make_nvp(TOPLEVEL_XML_ELEMENT, *this);
}

void FileLog::load(const std::string& filename)
{
  std::ifstream ifs(filename);
  boost::archive::xml_iarchive ia(ifs);
  ia >> boost::serialization::make_nvp(TOPLEVEL_XML_ELEMENT, *this);
}

bool FileLog::isSupportedImage(const std::string& filename)
{
  using namespace boost::algorithm;
  return iends_with(filename, ".png") || iends_with(filename, ".jpg");
}

bool FileLog::isSupportedVideo(const std::string& filename)
{
  using namespace boost::algorithm;
  return iends_with(filename, ".avi");
}

bool FileLog::isSupportedFormat(const std::string& filename)
{
  return isSupportedImage(filename) || isSupportedVideo(filename);
}

FileLog FileLog::detect(const std::string& filename, const Parameters& parameters)
{
  if (parameters._nCrowns != 3 && parameters._nCrowns != 4)
    throw std::runtime_error("FileLog: unsupported number of crowns; can only be 3 or 4");
  if (isSupportedImage(filename))
    return detectImage(filename, parameters);
  if(isSupportedVideo(filename))
    return detectVideo(filename, parameters);
  throw std::runtime_error(std::string("FileLog: unsupported format for file ") + filename);
}

FileLog FileLog::detectImage(const std::string& filename, const cctag::Parameters& parameters)
{
  FileLog fileLog(filename, parameters);
  CCTagMarkersBank bank(parameters._nCrowns);

  cv::Mat src, gray;
  src = cv::imread(filename);
  if (src.empty())
    throw std::runtime_error(std::string("FileLog: unable to read image file: ") + filename);
  cv::cvtColor(src, gray, CV_BGR2GRAY);
  
  auto frameLog = FrameLog::detect(0, gray, parameters, bank);
  fileLog.frameLogs.push_back(frameLog);
  return fileLog;
}

FileLog FileLog::detectVideo(const std::string& filename, const cctag::Parameters& parameters)
{
  FileLog fileLog(filename, parameters);
  CCTagMarkersBank bank(parameters._nCrowns);
  
  cv::VideoCapture video(filename.c_str());
  if (!video.isOpened())
    throw std::runtime_error("FileLog: unable to open video file: " + filename);
  
  const size_t lastFrame = video.get(CV_CAP_PROP_FRAME_COUNT);
  cv::Mat src, gray;
  
  for (size_t i = 0; i < lastFrame; ++i) {
    video >> src;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    auto frameLog = FrameLog::detect(i, gray, parameters, bank);
    fileLog.frameLogs.push_back(frameLog);
  }
  
  return fileLog;
}
