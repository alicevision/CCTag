#ifndef _CCTAG_2DTRANSFORM_HPP_
#define _CCTAG_2DTRANSFORM_HPP_

#include <vector>
#include <cctag/geometry/Ellipse.hpp>
#include <boost/foreach.hpp>
#include <Eigen/Core>

namespace cctag {
namespace viewGeometry {

void projectiveTransform( const Eigen::Matrix3f& tr, cctag::numerical::geometry::Ellipse& ellipse );

} // namespace viewGeometry
} // namespace cctag

#endif

