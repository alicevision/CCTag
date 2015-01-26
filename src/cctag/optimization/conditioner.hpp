#ifndef _ROM_CONDITIONER_HPP_
#define _ROM_CONDITIONER_HPP_

#include "../geometry/point.hpp"
#include "../algebra/matrix/Matrix.hpp"
#include "../statistic/statistic.hpp"
#include "../geometry/Ellipse.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/foreach.hpp>


namespace rom {
namespace numerical {
namespace optimization {


template<class C>
inline rom::numerical::BoundedMatrix3x3d conditionerFromPoints( const std::vector<C>& v )
{
	using namespace boost::numeric;
	rom::numerical::BoundedMatrix3x3d T;

	rom::numerical::BoundedVector3d m = rom::numerical::mean( v );
	rom::numerical::BoundedVector3d s = rom::numerical::stdDev( v, m );

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

inline rom::numerical::BoundedMatrix3x3d conditionerFromEllipse( const rom::numerical::geometry::Ellipse & ellipse )
{

	using namespace boost::numeric;
	rom::numerical::BoundedMatrix3x3d T;

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


inline void conditionerFromImage( const int c, const int r, const int f,  rom::numerical::BoundedMatrix3x3d & T, rom::numerical::BoundedMatrix3x3d & invT)
{
	using namespace boost::numeric;
	T(0,0) = 1.0 / f; T(0,1) = 0       ; T(0,2) = -c/(2.0 * f);
	T(1,0) = 0      ; T(1,1) = 1.0 / f ; T(1,2) = -r/(2.0 * f);
	T(2,0) = 0      ; T(2,1) = 0       ; T(2,2) = 1.0;

	invT(0,0) = f ; invT(0,1) = 0   ; invT(0,2) = c / 2.0;
	invT(1,0) = 0 ; invT(1,1) = f   ; invT(1,2) = r / 2.0;
	invT(2,0) = 0 ; invT(2,1) = 0   ; invT(2,2) = 1.0;

}


inline void condition(rom::Point2dN<double> & pt, const rom::numerical::BoundedMatrix3x3d & mT)
{
	using namespace boost::numeric;
	rom::numerical::BoundedVector3d cPt = ublas::prec_prod(mT,pt);
	BOOST_ASSERT( cPt(2) );
	pt.setX( cPt(0)/cPt(2) );
	pt.setY( cPt(1)/cPt(2) );
}


inline void condition(std::vector<rom::Point2dN<double> > & pts, const rom::numerical::BoundedMatrix3x3d & mT)
{
	using namespace boost::numeric;
	BOOST_FOREACH(rom::Point2dN<double> & pt, pts)
	{
		condition(pt, mT);
	}
}

}
}
}

#endif

