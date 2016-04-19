#pragma once

#include <math.h>
#include <vector>
#include <opencv/cv.h>
#include <boost/serialization/nvp.hpp>
#include "cctag/Detection.hpp"
#include "cctag/Params.hpp"

// Contains cctag info that is compared during regression testing.
struct DetectedTag
{
  static constexpr float POSITION_EPSILON = 0.5;  // sub-pixel accuracy claimed in the paper
  int id, status;
  double x, y, quality;
  
  // NB: we DO want implicit conversion for easy conversion of containers of CCTags
  DetectedTag(const CCTag& marker) :
    id(marker.id()), status(marker.getStatus()),
    x(marker.x()), y(marker.y()), quality(marker.quality())
  { }
  
  template<typename Archive>
  void serialize(Archive& ar, const unsigned)
  {
    ar & BOOST_SERIALIZATION_NVP(id);
    ar & BOOST_SERIALIZATION_NVP(status);
    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
    ar & BOOST_SERIALIZATION_NVP(quality);
  }
};

struct DetectionLog
{
  std::string filename;
  cctag::Parameters parameters;
  size_t frame;
  float elapsedTime;
  std::vector<DetectedTag> tags;
  
  DetectionLog(const std::string& filename, const cctag::Parameters& parameters,
    size_t frame, float elapsedTime, const CCTag::List& markers) :
    filename(filename), parameters(parameters), frame(frame), elapsedTime(elapsedTime),
    tags(markers.begin(), markers.end())
  { }
  
  template<typename Archive>
  void serialize(Archive& ar, const unsigned)
  {
    ar & BOOST_SERIALIZATION_NVP(filename);
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(frame);
    ar & BOOST_SERIALIZATION_NVP(elapsedTime);
    ar & BOOST_SERIALIZATION_NVP(tags);
  }
};
