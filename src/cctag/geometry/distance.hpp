#ifndef _ROM_DISTANCE_HPP_
#define _ROM_DISTANCE_HPP_

#include "Ellipse.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/units/cmath.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/pow.hpp>

namespace rom {
namespace numerical {

namespace ublas = boost::numeric::ublas;

template<class T>
inline double distancePoints2D( const T& p1, const T& p2 ) // TODO modifier les accès, considérer p1, p2 comme des bounded_vector
{
	return std::sqrt( (double)boost::math::pow<2>( p2.x() - p1.x() ) +
	                  boost::math::pow<2>( p2.y() - p1.y() ) );
}

template<class T>
inline double powDistancePoints2D( const T& p1, const T& p2 ) // TODO modifier les accès, considérer p1, p2 comme des bounded_vector
{
	return boost::math::pow<2>( p2.x() - p1.x() ) +
	       boost::math::pow<2>( p2.y() - p1.y() );
}

template<class T>
inline double distancePoints3D( const T& p1, const T& p2 ) // TODO modifier les accès, considérer p1, p2 comme des bounded_vector
{
	return std::sqrt( (double)( p2.x() - p1.x() ) * ( p2.x() - p1.x() ) +
	                  ( p2.y() - p1.y() ) * ( p2.y() - p1.y() ) +
	                  ( p2.z() - p1.z() ) * ( p2.z() - p1.z() ) );
}

// Compute (point-polar) distance between a point and an ellipse represented by its 3x3 matrix.
inline double distancePointEllipse( const ublas::bounded_vector<double, 3>& p, const ublas::bounded_matrix<double, 3, 3> & Q, const double f )
{
	ublas::bounded_vector<double, 6> aux( 6 );

	aux( 0 ) = p( 0 ) * p( 0 );
	aux( 1 ) = 2 * p( 0 ) * p( 1 );
	aux( 2 ) = 2* f* p( 0 );
	aux( 3 ) = p( 1 ) * p( 1 );
	aux( 4 ) = 2* f* p( 1 );
	aux( 5 ) = f * f;

	//sdist = ([pts(:,1).*pts(:,1) 2*pts(:,1).*pts(:,2) pts(:,2).*pts(:,2) 2*f*pts(:,1) 2*f*pts(:,2) f*f*ones(n,1)]*Q([1;2;5;7;8;9])).^2./((pts*Q([1;2;3])).^2+(pts*Q([2;5;8])).^2);
	double tmp1  = p( 0 ) * Q( 0, 0 ) + p( 1 ) * Q( 0, 1 ) + p( 2 ) * Q( 0, 2 );
	double tmp2  = p( 0 ) * Q( 0, 1 ) + p( 1 ) * Q( 1, 1 ) + p( 2 ) * Q( 1, 2 );
	double denom = tmp1 * tmp1 + tmp2 * tmp2;

	ublas::bounded_vector<double, 6> qL;
	qL( 0 ) = Q( 0, 0 ) ; qL( 1 ) = Q( 0, 1 ) ; qL( 2 ) = Q( 0, 2 ) ;
	qL( 3 ) = Q( 1, 1 ) ; qL( 4 ) = Q( 1, 2 ) ;
	qL( 5 ) = Q( 2, 2 );
	return boost::math::pow<2>( ublas::inner_prod( aux, qL ) ) / denom;
}

inline double distancePointEllipse( const ublas::bounded_vector<double, 3>& p, const geometry::Ellipse& q, const double f )
{
	const ublas::bounded_matrix<double, 3, 3>& Q = q.matrix();

	return distancePointEllipse( p, Q, f );
}

// Compute the distance between points and an ellipse
//template<class T>
inline void distancePointEllipse( std::vector<double>& dist, const std::vector<ublas::bounded_vector<double, 3> >& pts, const geometry::Ellipse& q, const double f )
{
	dist.clear();
	dist.reserve( pts.size() );
	typedef ublas::bounded_vector<double, 3> Point;

	BOOST_FOREACH( const Point &p, pts )
	{
		dist.push_back( distancePointEllipse( p, q, f ) );
	}
}

}
}

#endif
