#ifndef _POPART_VISION_MARKER_THINNING_HPP_
#define _POPART_VISION_MARKER_THINNING_HPP_

#include <opencv/cv.h>
#include <opencv2/core/types_c.h>
#include <boost/progress.hpp>
#include <iostream>


namespace popart {
namespace img {

void thin( IplImage* inout, IplImage* temp );

void imageIter( IplImage* in, IplImage* out, int* lut );

}
}

#endif
