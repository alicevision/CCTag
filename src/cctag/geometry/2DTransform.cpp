#include <cctag/geometry/2DTransform.hpp>

namespace cctag {

namespace viewGeometry {

void projectiveTransform( const Eigen::Matrix3f& tr, cctag::numerical::geometry::Ellipse& ellipse )
{
  ellipse.setMatrix(tr.transpose() * ellipse.matrix() * tr);
}

}
}
