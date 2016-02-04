#ifndef VISION_CCTAG_CVDRAW_HPP_
#define VISION_CCTAG_CVDRAW_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

// #include <opencv2/core/types_c.h>

#include <boost/foreach.hpp>

#include <vector>

namespace cctag {

void drawMarkerOnImage( IplImage* simg, const CCTag& marker );

void drawMarkersOnImage( IplImage* simg, const CCTag::Vector& markers );

} // namespace cctag

#endif
