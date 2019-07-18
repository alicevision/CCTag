/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>
#include <boost/serialization/nvp.hpp>
#include "cctag/Detection.hpp"
#include "cctag/Params.hpp"

// Contains cctag info that is compared during regression testing.
struct DetectedTag
{
  int id, status;
  float x, y, quality;
  
  template<typename Archive>
  void serialize(Archive& ar, const unsigned)
  {
    ar & BOOST_SERIALIZATION_NVP(id);
    ar & BOOST_SERIALIZATION_NVP(status);
    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
    ar & BOOST_SERIALIZATION_NVP(quality);
  }

  DetectedTag() = default;
  
  // NB: we DO want implicit conversion for easy conversion of containers of CCTags
  explicit DetectedTag(const cctag::CCTag& marker) :
    id(marker.id()), status(marker.getStatus()),
    x(marker.x()), y(marker.y()), quality(marker.quality())
  { }
};

struct FrameLog
{
  size_t frame;
  float elapsedTime;
  std::vector<DetectedTag> tags;
  
  template<typename Archive>
  void serialize(Archive& ar, const unsigned)
  {
    ar & BOOST_SERIALIZATION_NVP(frame);
    ar & BOOST_SERIALIZATION_NVP(elapsedTime);
    ar & BOOST_SERIALIZATION_NVP(tags);
  }
  
  FrameLog() = default;
  
  FrameLog(size_t frame, float elapsedTime, const cctag::CCTag::List& markers) :
    frame(frame), elapsedTime(elapsedTime), tags(markers.begin(), markers.end())
  { }

  static FrameLog detect(size_t frame, const cv::Mat& src, const cctag::Parameters& parameters,
    const cctag::CCTagMarkersBank& bank);
};

struct FileLog
{
  std::string filename;
  cctag::Parameters parameters;
  std::vector<FrameLog> frameLogs;
  
  template<typename Archive>
  void serialize(Archive& ar, const unsigned)
  {
    ar & BOOST_SERIALIZATION_NVP(filename);
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(frameLogs);
  }
  
  FileLog() = default;

  FileLog(const std::string& filename, const cctag::Parameters& parameters) :
    filename(filename), parameters(parameters)
  { }
  
  void save(const std::string& filename);
  void load(const std::string& filename);
  
  static bool isSupportedFormat(const std::string& filename);
  static FileLog detect(const std::string& filename, const cctag::Parameters& parameters);
  
private:
  static bool isSupportedImage(const std::string& filename);
  static bool isSupportedVideo(const std::string& filename);
  static FileLog detectImage(const std::string& filename, const cctag::Parameters& parameters);
  static FileLog detectVideo(const std::string& filename, const cctag::Parameters& parameters);
};
