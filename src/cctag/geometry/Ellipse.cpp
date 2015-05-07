#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/progBase/exceptions.hpp>

#include <boost/math/special_functions/sign.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

#include <algorithm>
#include <cmath>

namespace cctag {
namespace numerical {
namespace geometry {

using namespace boost::numeric::ublas;

Ellipse::Ellipse( const bounded_matrix<double, 3, 3>& matrix )
{
	_matrix = matrix;
	computeParameters();
}

Ellipse::Ellipse( const Point2dN<double>& center, const double a, const double b, const double angle )
{
	init( center, a, b, angle );
}

void Ellipse::init( const Point2dN<double>& center, const double a, const double b, const double angle )
{
	if( a < 0.0 || b < 0.0 )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}

	_center = center;
	_a      = a;
	_b      = b;
	_angle  = angle;

	computeMatrix();
}

void Ellipse::setMatrix( const bounded_matrix<double, 3, 3>& matrix )
{
	_matrix = matrix;
	computeParameters();
}

void Ellipse::setParameters( const Point2dN<double>& center, const double a, const double b, const double angle )
{
	if( a < 0.0 || b < 0.0 )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_center = center;
	_a      = a;
	_b      = b;
	_angle  = angle;
	computeMatrix();
}

void Ellipse::setCenter( const Point2dN<double>& center )
{
	_center = center;
	computeMatrix();
}

void Ellipse::setA( const double a )
{
	if( a < 0.0 )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_a = a;
	computeMatrix();
}

void Ellipse::setB( const double b )
{
	if( b < 0.0 )
	{
		CCTAG_THROW( exception::Bug()
			<< exception::dev( "Semi axes must be real positive!" ) );
	}
	_b = b;
	computeMatrix();
}

void Ellipse::setAngle( const double angle )
{
	_angle = angle;
	computeMatrix();
}

Ellipse Ellipse::transform(const Matrix& mT) const
{
	using namespace boost::numeric::ublas;
	const Matrix a = boost::numeric::ublas::prec_prod( boost::numeric::ublas::trans(mT), _matrix );
	const Matrix mET = boost::numeric::ublas::prec_prod( a, mT );
	return Ellipse( mET );
}

void Ellipse::computeParameters()
{
	bounded_vector<double, 6> par;
	par( 0 ) = _matrix( 0, 0 );
	par( 1 ) = 2.0 * _matrix( 0, 1 );
	par( 2 ) = _matrix( 1, 1 );
	par( 3 ) = 2 * _matrix( 0, 2 );
	par( 4 ) = 2 * _matrix( 1, 2 );
	par( 5 ) = _matrix( 2, 2 );

	const double thetarad    = 0.5 * std::atan2( par( 1 ), par( 0 ) - par( 2 ) );
	const double cost        = std::cos( thetarad );
	const double sint        = std::sin( thetarad );
	const double sin_squared = sint * sint;
	const double cos_squared = cost * cost;
	const double cos_sin     = sint * cost;

	const double Ao  = par( 5 );
	const double Au  = par( 3 ) * cost + par( 4 ) * sint;
	const double Av  = -par( 3 ) * sint + par( 4 ) * cost;
	const double Auu = par( 0 ) * cos_squared + par( 2 ) * sin_squared + par( 1 ) * cos_sin;
	const double Avv = par( 0 ) * sin_squared + par( 2 ) * cos_squared - par( 1 ) * cos_sin;

	if( Auu == 0 || Avv == 0 )
	{
		_center = Point2dN<double>( 0.0, 0.0 );
		_a      = 0.0;
		_b      = 0.0;
		_angle  = 0.0;
	}
	else
	{
		const double tuCentre = -Au / ( 2.0 * Auu );
		const double tvCentre = -Av / ( 2.0 * Avv );
		const double wCentre  = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;

		_center = Point2dN<double>( tuCentre * cost - tvCentre * sint, tuCentre * sint + tvCentre * cost );

		const double Ru = -wCentre / Auu;
		const double Rv = -wCentre / Avv;

		const double aAux = std::sqrt( std::abs( Ru ) ) * boost::math::sign( Ru );
		const double bAux = std::sqrt( std::abs( Rv ) ) * boost::math::sign( Rv );

		if( aAux < 0.0 || bAux < 0.0 )
		{
			CCTAG_THROW( exception::Bug()
				<< exception::dev( "Semi axes must be real positive!" ) );
		}

		_a     = aAux;
		_b     = bAux;
		_angle = thetarad;
	}
}

void Ellipse::computeMatrix()
{
	bounded_matrix<double, 3, 3> tmp;
	tmp( 0, 0 ) = std::cos( _angle ); tmp( 0, 1 ) = -std::sin( _angle ); tmp( 0, 2 ) = _center.x();
	tmp( 1, 0 ) = std::sin( _angle ); tmp( 1, 1 ) =  std::cos( _angle ); tmp( 1, 2 ) = _center.y();
	tmp( 2, 0 ) =             0.0; tmp( 2, 1 ) =              0.0; tmp( 2, 2 ) =        1.0;

	bounded_matrix<double, 3, 3> tmpInv;
	diagonal_matrix<double> diag( 3, 3 );
	diag( 0, 0 ) =  1.0 / ( _a * _a );
	diag( 1, 1 ) =  1.0 / ( _b * _b );
	diag( 2, 2 ) = -1.0;

	if( invert( tmp, tmpInv ) )
	{
		_matrix = prec_prod( diag, tmpInv );
		_matrix = prec_prod( trans( tmpInv ), _matrix );
	}
	else
	{
		CCTAG_THROW( exception::Bug()
				<< exception::dev( "Singular matrix!" ) );
	}
}

void scale(const Ellipse & ellipse, Ellipse & rescaleEllipse, double scale)
{
	rescaleEllipse.setCenter(Point2dN<double>( ellipse.center().x() * scale, ellipse.center().y() * scale ));
	rescaleEllipse.setA(ellipse.a() * scale);
	rescaleEllipse.setB(ellipse.b() * scale);
	rescaleEllipse.setAngle(ellipse.angle());
}

std::ostream& operator<<(std::ostream& os, const Ellipse& e)
{
	os << "e = [ " << e.matrix()(0,0) << " " << e.matrix()(0,1) << " " << e.matrix()(0,2) << " ; "
				   << e.matrix()(1,0) << " " << e.matrix()(1,1) << " " << e.matrix()(1,2) << " ; "
				   << e.matrix()(2,0) << " " << e.matrix()(2,1) << " " << e.matrix()(2,2) << " ] ";
	return os;
}

}
}
}
