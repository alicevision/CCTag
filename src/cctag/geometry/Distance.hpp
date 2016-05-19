#ifndef _CCTAG_DISTANCE_HPP_
#define _CCTAG_DISTANCE_HPP_

#include <cctag/geometry/Ellipse.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/units/cmath.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <Eigen/Dense>
#include <tbb/tbb.h>

namespace cctag {
namespace numerical {

template<class T, class U>
inline float distancePoints2D( const T& p1, const U& p2 ) // TODO modifier les accès, considérer p1, p2 comme des bounded_vector
{
	return std::sqrt( (float)boost::math::pow<2>( p2.x() - p1.x() ) +
	                  boost::math::pow<2>( p2.y() - p1.y() ) );
}

float distancePointEllipseScalar(const Eigen::Vector3f& p, const Eigen::Matrix3f& Q, const float f);

template <class T>
inline float distancePointEllipse( const T & p, const geometry::Ellipse& q, const float f )
{
  return distancePointEllipseScalar(p.template cast<float>(), q.matrix(), f);
}

// Compute the distance between points and an ellipse
//template<class T>
void distancePointEllipse( std::vector<float>& dist, const std::vector<Eigen::Vector3f>& pts, const geometry::Ellipse& q, const float f );

}
}

#endif
