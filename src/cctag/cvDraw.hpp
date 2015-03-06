#ifndef VISION_CCTAG_CVDRAW_HPP_
#define VISION_CCTAG_CVDRAW_HPP_

#include "CCTag.hpp"

#include <cctag/geometry/Ellipse.hpp>

#include <opencv2/core/types_c.h>

#include <boost/foreach.hpp>

#include <vector>

namespace cctag {
namespace vision {
namespace marker {

void drawMarkerOnImage( IplImage* simg, const CCTag& marker );

void drawMarkersOnImage( IplImage* simg, const CCTag::Vector& markers );

}
}
}

#endif
