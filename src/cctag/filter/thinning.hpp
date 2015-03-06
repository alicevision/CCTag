#ifndef VISION_MARKER_THINNING_HPP_
#define VISION_MARKER_THINNING_HPP_

#include <opencv/cv.h>
#include <opencv2/core/types_c.h>
#include <boost/progress.hpp>
#include <iostream>


namespace cctag {
namespace img {

void thin( IplImage* inout, IplImage* temp );

void imageIter( IplImage* in, IplImage* out, int* lut );

}
}

#endif
