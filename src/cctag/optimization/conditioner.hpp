#ifndef _CCTAG_CONDITIONER_HPP_
#define _CCTAG_CONDITIONER_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/foreach.hpp>


namespace cctag {
namespace numerical {
namespace optimization {


template<class C>
inline cctag::numerical::BoundedMatrix3x3d conditionerFromPoints( const std::vector<C>& v )
{
	using namespace boost::numeric;
	cctag::numerical::BoundedMatrix3x3d T;

	cctag::numerical::BoundedVector3d m = cctag::numerical::mean( v );
	cctag::numerical::BoundedVector3d s = cctag::numerical::stdDev( v, m );

	if( s( 0 ) == 0 )
		s( 0 )++;
	if( s( 1 ) == 0 )
		s( 1 )++;

	static const double sqrt2 = std::sqrt( 2.0 );
	T( 0, 0 ) = sqrt2 / s( 0 );
	T( 0, 1 ) = 0;
	T( 0, 2 ) = -sqrt2* m( 0 ) / s( 0 );

	T( 1, 0 ) = 0;
	T( 1, 1 ) = sqrt2 / s( 1 );
	T( 1, 2 ) = -sqrt2* m( 1 ) / s( 1 );

	T( 2, 0 ) = 0;
	T( 2, 1 ) = 0;
	T( 2, 2 ) = 1;

	return T;
}

inline cctag::numerical::BoundedMatrix3x3d conditionerFromEllipse( const cctag::numerical::geometry::Ellipse & ellipse )
{

	using namespace boost::numeric;
	cctag::numerical::BoundedMatrix3x3d T;

	static const double sqrt2 = std::sqrt( 2.0 );
	static const double meanAB = (ellipse.a()+ellipse.b())/2.0;

	//[ 2^(1/2)/a,         0, -(2^(1/2)*x0)/a]
//[         0, 2^(1/2)/a, -(2^(1/2)*y0)/a]
//[         0,         0,               1]

	T( 0, 0 ) = sqrt2 / meanAB;
	T( 0, 1 ) = 0;
	T( 0, 2 ) = -sqrt2* ellipse.center().x() / meanAB;

	T( 1, 0 ) = 0;
	T( 1, 1 ) = sqrt2 / meanAB;
	T( 1, 2 ) = -sqrt2* ellipse.center().y() / meanAB;

	T( 2, 0 ) = 0;
	T( 2, 1 ) = 0;
	T( 2, 2 ) = 1;

	return T;
}


inline void conditionerFromImage( const int c, const int r, const int f,  cctag::numerical::BoundedMatrix3x3d & T, cctag::numerical::BoundedMatrix3x3d & invT)
{
	using namespace boost::numeric;
	T(0,0) = 1.0 / f; T(0,1) = 0       ; T(0,2) = -c/(2.0 * f);
	T(1,0) = 0      ; T(1,1) = 1.0 / f ; T(1,2) = -r/(2.0 * f);
	T(2,0) = 0      ; T(2,1) = 0       ; T(2,2) = 1.0;

	invT(0,0) = f ; invT(0,1) = 0   ; invT(0,2) = c / 2.0;
	invT(1,0) = 0 ; invT(1,1) = f   ; invT(1,2) = r / 2.0;
	invT(2,0) = 0 ; invT(2,1) = 0   ; invT(2,2) = 1.0;

}

template <class T>
inline void condition(T & point, const cctag::numerical::BoundedMatrix3x3d & mTransformation)
{
  using namespace boost::numeric;
  cctag::numerical::BoundedVector3d conditionedPoint = ublas::prec_prod(mTransformation,point);
  BOOST_ASSERT( conditionedPoint(2) );
  point.setX( conditionedPoint(0)/conditionedPoint(2) );
  point.setY( conditionedPoint(1)/conditionedPoint(2) );
}

template <class T>
inline void condition(std::vector<T> & points, const cctag::numerical::BoundedMatrix3x3d & mTransformation)
{
  using namespace boost::numeric;
  BOOST_FOREACH(T & point, points)
  {
    condition(point, mTransformation);
  }
}

}
}
}

#endif

