#ifndef _POPART_VISION_CCTAG_CVDRAW_HPP_
#define _POPART_VISION_CCTAG_CVDRAW_HPP_

#include "CCTag.hpp"

#include <cctag/geometry/Ellipse.hpp>

#include <opencv2/core/types_c.h>

#include <boost/foreach.hpp>

#include <vector>

namespace popart {
namespace vision {
namespace marker {

void drawMarkerOnImage( IplImage* simg, const CCTag& marker );

void drawMarkersOnImage( IplImage* simg, const CCTag::Vector& markers );

}
}
}

#endif
