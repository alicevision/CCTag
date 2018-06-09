/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_DISTANCE_HPP_
#define _CCTAG_DISTANCE_HPP_

#include <cctag/geometry/Ellipse.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <Eigen/Dense>

namespace cctag {
namespace numerical {

template<class T, class U>
inline float distancePoints2D( const T& p1, const U& p2 )
{
	return std::sqrt( (float)boost::math::pow<2>( p2.x() - p1.x() ) +
	                  (float)boost::math::pow<2>( p2.y() - p1.y() ) );
}

float distancePointEllipseScalar(const Eigen::Vector3f& p, const Eigen::Matrix3f& Q);

template <class T>
inline float distancePointEllipse( const T & p, const geometry::Ellipse& q)
{
  return distancePointEllipseScalar(p.template cast<float>(), q.matrix());
}

// Compute the distance between points and an ellipse
void distancePointEllipse( std::vector<float>& dist, const std::vector<Eigen::Vector3f>& pts, const geometry::Ellipse& q);

}
}

#endif
