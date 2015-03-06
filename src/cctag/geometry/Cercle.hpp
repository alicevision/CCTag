#ifndef _CCTAG_NUMERICAL_CERCLE_HPP_
#define _CCTAG_NUMERICAL_CERCLE_HPP_

#include "Ellipse.hpp"

#include <cctag/geometry/point.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/global.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>

#include <cmath>

namespace cctag {
namespace numerical {
namespace geometry {

using namespace boost::numeric::ublas;

class Cercle : public Ellipse
{
public:

	Cercle() : Ellipse() {}

	Cercle( const double r );

	Cercle( const Point2dN<double>& center, const double r );
	virtual ~Cercle();

	template <typename T>
	Cercle( const Point2dN<T> & p1, const Point2dN<T> & p2, const Point2dN<T> & p3 )
	{
		const T x1 = p1.x();
		const T y1 = p1.y();

		const T x2 = p2.x();
		const T y2 = p2.y();

		const T x3 = p3.x();
		const T y3 = p3.y();

		const T det = ( x1 - x2 ) * ( y1 - y3 ) - ( y1 - y2 ) * ( x1 - x3 );

		if( det == 0 )
		{
			///@todo
		}

		bounded_matrix<double, 2, 2> A( 2, 2 );

		A( 0, 0 ) = (double) x2 - x1;
		A( 0, 1 ) = (double) y2 - y1;
		A( 1, 0 ) = (double) x3 - x1;
		A( 1, 1 ) = (double) y3 - y1;

		bounded_vector<double, 2> bb( 2 );

		bb( 0 ) = (double) ( x1 + x2 ) / 2 * ( x2 - x1 ) + ( y1 + y2 ) / 2 * ( y2 - y1 );
		bb( 1 ) = (double) ( x1 + x3 ) / 2 * ( x3 - x1 ) + ( y1 + y3 ) / 2 * ( y3 - y1 );

		bounded_matrix<double, 2, 2> AInv( 2, 2 );

		cctag::numerical::invert_2x2( A, AInv );

		bounded_vector<double, 2> aux = prec_prod( AInv, bb );

		double xc = aux( 0 );
		double yc = aux( 1 );

		double r = sqrt( ( x1 - xc ) * ( x1 - xc ) + ( y1 - yc ) * ( y1 - yc ) );

		ROM_COUT_LILIAN( " xc = " << xc << " yc = " << yc << " r = " << r );

		Point2dN<double> c( xc, yc );

		Ellipse::init( c, r, r, 0.0 );

		ROM_COUT_LILIAN( "center x = " << _center.x() << "  y = " << center().y() << " a = " << a() << " b = " << b() );

	}

private:
};

}
}
}

#endif

