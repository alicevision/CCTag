#ifndef _CCTAG_CONDITIONER_HPP_
#define _CCTAG_CONDITIONER_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/foreach.hpp>


namespace cctag {
namespace numerical {
namespace optimization {

#if 0
template<class C>
inline Eigen::Matrix3f conditionerFromPoints( const std::vector<C>& v )
{
	using namespace boost::numeric;
	Eigen::Matrix3f T;

	cctag::numerical::BoundedVector3d m = cctag::numerical::mean( v );
	cctag::numerical::BoundedVector3d s = cctag::numerical::stdDev( v, m );

	if( s( 0 ) == 0 )
		s( 0 )++;
	if( s( 1 ) == 0 )
		s( 1 )++;

	static const float sqrt2 = std::sqrt( 2.0 );
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
#endif

inline Eigen::Matrix3f conditionerFromEllipse( const cctag::numerical::geometry::Ellipse & ellipse )
{
	Eigen::Matrix3f res;

	static const float sqrt2 = std::sqrt( 2.f );
	static const float meanAB = (ellipse.a()+ellipse.b())/2.f;

	//[ 2^(1/2)/a,         0, -(2^(1/2)*x0)/a]
        //[         0, 2^(1/2)/a, -(2^(1/2)*y0)/a]
        //[         0,         0,               1]

	res( 0, 0 ) = sqrt2 / meanAB;
	res( 0, 1 ) = 0.f;
	res( 0, 2 ) = -sqrt2* ellipse.center().x() / meanAB;

	res( 1, 0 ) = 0.f;
	res( 1, 1 ) = sqrt2 / meanAB;
	res( 1, 2 ) = -sqrt2* ellipse.center().y() / meanAB;

	res( 2, 0 ) = 0.f;
	res( 2, 1 ) = 0.f;
	res( 2, 2 ) = 1.f;

	return res;
}


inline void conditionerFromImage( const int c, const int r, const int f,  Eigen::Matrix3f & trans, Eigen::Matrix3f & invTrans)
{
	using namespace boost::numeric;
	trans(0,0) = 1.f / f  ; trans(0,1) = 0.f       ; trans(0,2) = -c/(2.0 * f);
	trans(1,0) = 0.f      ; trans(1,1) = 1.f / f   ; trans(1,2) = -r/(2.0 * f);
	trans(2,0) = 0.f      ; trans(2,1) = 0.f       ; trans(2,2) = 1.f;

	invTrans(0,0) = f   ; invTrans(0,1) = 0.f   ; invTrans(0,2) = c / 2.0;
	invTrans(1,0) = 0.f ; invTrans(1,1) = f     ; invTrans(1,2) = r / 2.0;
	invTrans(2,0) = 0.f ; invTrans(2,1) = 0.f   ; invTrans(2,2) = 1.f;
}

template <class T>
inline void condition(T & point, const Eigen::Matrix3f & mTransformation)
{
  using namespace boost::numeric;
  const Eigen::Vector3f conditionedPoint = mTransformation*point;
  point.x() = conditionedPoint(0)/conditionedPoint(2);
  point.y() = conditionedPoint(1)/conditionedPoint(2);
}

template <class T>
inline void condition(std::vector<T> & points, const Eigen::Matrix3f & mTransformation)
{
  using namespace boost::numeric;
  for(auto & point: points)
    condition(point, mTransformation);
}

}
}
}

#endif

