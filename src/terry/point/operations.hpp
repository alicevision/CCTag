#ifndef _TERRY_POINT_OPERATIONS_HPP_
#define _TERRY_POINT_OPERATIONS_HPP_

#include <boost/gil/utilities.hpp>


/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<T> operator*( const boost::gil::point2<T>& p, const double t ) { return boost::gil::point2<T >( p.x * t, p.y * t ); }
/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<T>& operator*=( boost::gil::point2<T>& p, const double t ) { p.x *= t; p.y *= t; return p; }
/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<T> operator*( const boost::gil::point2<T>& a, const boost::gil::point2<T>& b ) { return boost::gil::point2<T>( a.x * b.x, a.y * b.y ); }
/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<T>& operator*=( boost::gil::point2<T>& a, const boost::gil::point2<T>& b ) { a.x *= b.x; a.y *= b.y; return a; }
/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<T> operator/( const boost::gil::point2<T>& a, const boost::gil::point2<T>& b ) { return boost::gil::point2<T>( a.x / b.x, a.y / b.y ); }
/// \ingroup PointModel
template <typename T>
GIL_FORCEINLINE
boost::gil::point2<double> operator/( const double t, const boost::gil::point2<T>& p )
{
	boost::gil::point2<double> res( 0, 0 );
	if( p.x != 0 )
		res.x = t / p.x;
	if( p.y != 0 )
		res.y = t / p.y;
	return res;
}


#endif

