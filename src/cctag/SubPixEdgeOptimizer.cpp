#include <cctag/SubPixEdgeOptimizer.hpp>
#include <cctag/imageCut.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/progBase/exceptions.hpp>
#include <cctag/global.hpp>

#include <OptQNewton.h>
#include <newmat.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/pow.hpp>

#include <cmath>
#include <cstddef>
#include <ostream>

namespace cctag {

SubPixEdgeOptimizer::SubPixEdgeOptimizer( const cctag::ImageCut & line )
: Parent( 4, &SubPixEdgeOptimizer::subPix, NULL, this )
, _line( line )
{
	_a = ( line._stop.y() - line._start.y() ) / ( line._stop.x() - line._start.x() );
	_b = line._start.y() - _a * line._start.x();
}

void SubPixEdgeOptimizer::initSubPix( int ndim, NEWMAT::ColumnVector& x )
{
	if ( ndim != 4 )
	{
		BOOST_THROW_EXCEPTION( exception::Bug() << exception::dev() + "Unable to init minimizer!" );
	}

	x(1) = _widthContour;
	x(2) = _xp;
	x(3) = _imin;
	x(4) = _imax;
}

/**
 * Fonction de coût (fct à minimiser)
 * 
 * @param[int] n nombre de paramètres à minimiser
 * @param[in] x les paramètres à estimer
 * @param[out] fx un scalaire, le résultat
 * @param[out] result, un truc de optpp, je sais pas pour l'instant à quoi ça sert
 * @param[in] objPtr
 */
void SubPixEdgeOptimizer::subPix( int n, const NEWMAT::ColumnVector& x, double& fx, int& result, void *objPtr )
{
	using namespace OPTPP;

	if( n != 4 )
	{
		return;
	}

	This *this_ptr = static_cast<This*>( objPtr );

	const double normDir = cctag::numerical::distancePoints2D( this_ptr->_line._start, this_ptr->_line._stop );
	const double dirx = ( this_ptr->_line._stop.x() - this_ptr->_line._start.x() ) / normDir;
	const double diry = ( this_ptr->_line._stop.y() - this_ptr->_line._start.y() ) / normDir;

	const double width = x(1);
	const double p0x = x(2);
	const double p0y = this_ptr->_a * p0x + this_ptr->_b;

	const double imin = x(3);
	const double imax = x(4);

	double res = 0;

	const std::size_t signalLength = this_ptr->_line._imgSignal.size();
	const double kx = ( this_ptr->_line._stop.x() - this_ptr->_line._start.x() ) / ( signalLength - 1.0 );
	const double ky = ( this_ptr->_line._stop.y() - this_ptr->_line._start.y() ) / ( signalLength - 1.0 );
	double pindx = this_ptr->_line._start.x();
	double pindy = this_ptr->_line._start.y();

	const boost::numeric::ublas::vector<double> & sig = this_ptr->_line._imgSignal;

	for( std::size_t i = 0 ; i < signalLength ; ++i )
	{
		const double t = ( ( pindx - p0x ) * dirx + ( pindy - p0y ) * diry ) / width;

		// ft: signal modelization
		double ft;
		if ( t < -1.0 )
		{
			ft = imin;
		}
		else if ( t > 1.0 )
		{
			ft = imax;
		}
		else
		{
			// inside the slope
			const double gt = 0.5 + 1.0 / boost::math::constants::pi<double>() * ( t * std::sqrt( 1.0 - t * t ) + std::asin( t ) );
			ft = gt * ( imax - imin ) + imin;
		}
		// minimization of the distance between signal and parametric model (least square sense).
		res += boost::math::pow<2>( sig(i) - ft );

		pindx += kx;
		pindy += ky;
	}

	fx = res;
	result = NLPFunction;
}

Point2dN<double> SubPixEdgeOptimizer::operator()( const double widthContour, const double xp, const double imin, const double imax )
{
	using namespace OPTPP;
	using namespace NEWMAT;

	Point2dN<double> res;
	//  Create a Nonlinear problem object
	_widthContour = widthContour;
	_xp = xp;
	_imin = imin;
	_imax = imax;

	OptQNewton objfcn( this );

	objfcn.setSearchStrategy( TrustRegion );
	objfcn.setMaxFeval( 200 );
	objfcn.setFcnTol( 1.0e-4 );

	objfcn.optimize();

	objfcn.printStatus("Solution from quasi-newton");
	objfcn.cleanup();

	//#ifdef REG_TEST
	ColumnVector x_sol = getXc();
	double f_sol = getF();

	// CCTAG_TCOUT( "Solution :" ); //don't delete.

	// CCTAG_TCOUT( "width : " << x_sol(1) ); //don't delete.
	// CCTAG_TCOUT( "xp : " << x_sol(2) ); //don't delete.
	// CCTAG_TCOUT( "imin : " << x_sol(3) ); //don't delete.
	// CCTAG_TCOUT( "imax : " << x_sol(4) ); //don't delete.

	// Point raffiné à retourner :
	res.setX( x_sol(2) );
	res.setY( _a * x_sol(2) + _b );

	//CCTAG_TCOUT( "p0raffine : (" << res ); //don't delete.

	objfcn.cleanup();

	return res;
}

} // namespace cctag
