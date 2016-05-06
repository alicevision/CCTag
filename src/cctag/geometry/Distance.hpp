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
        using Vector6f = Eigen::Matrix<float, 6, 1>;
	Vector6f aux( 6 );

        float x = p.x();
        float y = p.y();
        
	aux( 0 ) = x*x;
	aux( 1 ) = 2 * x * y;
	aux( 2 ) = 2* x;
	aux( 3 ) = y * y;
	aux( 4 ) = 2* y;
	aux( 5 ) = 1;

	float tmp1  = p( 0 ) * Q( 0, 0 ) + p( 1 ) * Q( 0, 1 ) + p( 2 ) * Q( 0, 2 );
	float tmp2  = p( 0 ) * Q( 0, 1 ) + p( 1 ) * Q( 1, 1 ) + p( 2 ) * Q( 1, 2 );
	float denom = tmp1 * tmp1 + tmp2 * tmp2;

	Vector6f qL(6);
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
        const size_t n = pts.size();
	dist.resize( n );
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < n; ++i)
          dist[i] = distancePointEllipse( pts[i], q, f );
}

}
}

#endif
