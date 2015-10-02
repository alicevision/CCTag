#ifndef VISION_MARKER_THINNING_HPP_
#define VISION_MARKER_THINNING_HPP_

#include <opencv/cv.h>
#include <opencv2/core/types_c.h>
#include <boost/progress.hpp>
#include <iostream>


namespace cctag {

void thin( cv::Mat & inout, cv::Mat & temp );

void imageIter( cv::Mat & in, cv::Mat & out, int* lut );

}

#endif
