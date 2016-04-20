#pragma once

#include <vector>
#include <opencv/cv.h>
#include <boost/serialization/nvp.hpp>
#include "cctag/Detection.hpp"
#include "cctag/Params.hpp"

// Contains cctag info that is compared during regression testing.
struct DetectedTag
{
  static float EQUALITY_EPSILON;  // configurable
  
  int id, status;
  double x, y, quality;
  
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
  DetectedTag(const cctag::CCTag& marker) :
    id(marker.id()), status(marker.getStatus()),
    x(marker.x()), y(marker.y()), quality(marker.quality())
  { }
  
  bool operator==(const DetectedTag&) const;

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
  
  bool operator==(const FrameLog&) const;

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
  
  static void save(const std::string& filename, const FileLog& fileLog);
  static FileLog load(const std::string& filename);
  static bool isSupportedFormat(const std::string& filename);
  static FileLog detect(const std::string& filename, const cctag::Parameters& parameters);
  
private:
  static bool isSupportedImage(const std::string& filename);
  static bool isSupportedVideo(const std::string& filename);
  static FileLog detectImage(const std::string& filename, const cctag::Parameters& parameters);
  static FileLog detectVideo(const std::string& filename, const cctag::Parameters& parameters);
};
