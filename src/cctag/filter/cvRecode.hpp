#ifndef _CCTAG_CVRECODE_HPP_
#define _CCTAG_CVRECODE_HPP_

#include <opencv2/core/core.hpp>

void cvRecodedCanny(
  const cv::Mat & imgGraySrc,
  cv::Mat& imgCanny,
  cv::Mat& imgDX,
  cv::Mat& imgDY,
  double low_thresh,
  double high_thresh,
  int aperture_size );
#endif

