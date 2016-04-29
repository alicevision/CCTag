#ifndef _CCTAG_DISTANCE_HPP_
#define _CCTAG_DISTANCE_HPP_

#include <cctag/geometry/Ellipse.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/units/cmath.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <Eigen/Dense>

namespace cctag {
namespace numerical {

template<class T, class U>
inline float distancePoints2D( const T& p1, const U& p2 ) // TODO modifier les accès, considérer p1, p2 comme des bounded_vector
{
	return std::sqrt( (float)boost::math::pow<2>( p2.x() - p1.x() ) +
	                  boost::math::pow<2>( p2.y() - p1.y() ) );
}

// Compute (point-polar) distance between a point and an ellipse represented by its 3x3 matrix.
// TODO@lilian: f is always equal to 1, remove it
template <class T>
inline float distancePointEllipse(const T & p, const Eigen::Matrix3f& Q, const float f )
{
	Eigen::VectorXf aux( 6 );

        float x = p.x();
        float y = p.y();
        
	aux( 0 ) = x*x;
	aux( 1 ) = 2 * x * y;
	aux( 2 ) = 2* f* x;
	aux( 3 ) = y * y;
	aux( 4 ) = 2* f* y;
	aux( 5 ) = f * f;

	float tmp1  = p( 0 ) * Q( 0, 0 ) + p( 1 ) * Q( 0, 1 ) + p( 2 ) * Q( 0, 2 );
	float tmp2  = p( 0 ) * Q( 0, 1 ) + p( 1 ) * Q( 1, 1 ) + p( 2 ) * Q( 1, 2 );
	float denom = tmp1 * tmp1 + tmp2 * tmp2;

	Eigen::VectorXf qL(6);
	qL( 0 ) = Q( 0, 0 ) ; qL( 1 ) = Q( 0, 1 ) ; qL( 2 ) = Q( 0, 2 ) ;
	qL( 3 ) = Q( 1, 1 ) ; qL( 4 ) = Q( 1, 2 ) ;
	qL( 5 ) = Q( 2, 2 );
	return boost::math::pow<2>( aux.dot(qL) ) / denom;
}

template <class T>
inline float distancePointEllipse( const T & p, const geometry::Ellipse& q, const float f )
{
	const auto& Q = q.matrix();
	return distancePointEllipse( p, Q, f );
}

// Compute the distance between points and an ellipse
//template<class T>
inline void distancePointEllipse( std::vector<float>& dist, const std::vector<Eigen::Vector3f>& pts, const geometry::Ellipse& q, const float f )
{
	dist.clear();
	dist.reserve( pts.size() );
        for (const auto& p : pts)
          dist.push_back( distancePointEllipse( p, q, f ) );
}

}
}

#endif
