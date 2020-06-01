/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/SubPixEdgeOptimizer.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/utils/Exceptions.hpp>
#include <cctag/utils/Defines.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/pow.hpp>

#include <cmath>
#include <cstddef>
#include <ostream>

namespace cctag {

#if defined(WITH_OPTPP) && defined(SUBPIX_EDGE_OPTIM) // undefined. Deprecated

SubPixEdgeOptimizer::SubPixEdgeOptimizer( const cctag::ImageCut & line )
: Parent( 4, &SubPixEdgeOptimizer::subPix, NULL, this )
, _line( line )
{
	_a = ( line.stop().y() - line.start().y() ) / ( line.stop().x() - line.start().x() );
	_b = line.start().y() - _a * line.start().x();
}

void SubPixEdgeOptimizer::initSubPix( int ndim, NEWMAT::ColumnVector& x )
{
	if ( ndim != 4 )
	{
		throw exception::Bug("Unable to init minimizer!");
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

	const float normDir = cctag::numerical::distancePoints2D( this_ptr->_line.start(), this_ptr->_line.stop() );
	const float dirx = ( this_ptr->_line.stop().x() - this_ptr->_line.start().x() ) / normDir;
	const float diry = ( this_ptr->_line.stop().y() - this_ptr->_line.start().y() ) / normDir;

	const float width = x(1);
	const float p0x = x(2);
	const float p0y = this_ptr->_a * p0x + this_ptr->_b;

	const float imin = x(3);
	const float imax = x(4);

	float res = 0;

	const std::size_t signalLength = this_ptr->_line.imgSignal().size();
	const float kx = ( this_ptr->_line.stop().x() - this_ptr->_line.start().x() ) / ( signalLength - 1.f );
	const float ky = ( this_ptr->_line.stop().y() - this_ptr->_line.start().y() ) / ( signalLength - 1.f );
	float pindx = this_ptr->_line.start().x();
	float pindy = this_ptr->_line.start().y();

	const boost::numeric::ublas::vector<float> & sig = this_ptr->_line.imgSignal();

	for( std::size_t i = 0 ; i < signalLength ; ++i )
	{
		const float t = ( ( pindx - p0x ) * dirx + ( pindy - p0y ) * diry ) / width;

		// ft: signal modelization
		float ft;
		if ( t < -1.f )
		{
			ft = imin;
		}
		else if ( t > 1.f )
		{
			ft = imax;
		}
		else
		{
			// inside the slope
			const float gt = 0.5f + 1.f / boost::math::constants::pi<float>() * ( t * std::sqrt( 1.f - t * t ) + std::asin( t ) );
			ft = gt * ( imax - imin ) + imin;
		}
		// minimization of the distance between signal and parametric model (least square sense).
		res += boost::math::pow<2>( sig(i) - ft );

		pindx += kx;
		pindy += ky;
	}

	fx = (double)res;
	result = NLPFunction;
}

Point2d<Eigen::Vector3f> SubPixEdgeOptimizer::operator()( const float widthContour, const float xp, const float imin, const float imax )
{
	using namespace OPTPP;
	using namespace NEWMAT;

	Point2d<Eigen::Vector3f> res;
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

	objfcn.cleanup();

	//#ifdef REG_TEST
	ColumnVector x_sol = getXc();
	float f_sol = getF();

	// CCTAG_TCOUT( "Solution :" ); //don't delete.

	// CCTAG_TCOUT( "width : " << x_sol(1) ); //don't delete.
	// CCTAG_TCOUT( "xp : " << x_sol(2) ); //don't delete.
	// CCTAG_TCOUT( "imin : " << x_sol(3) ); //don't delete.
	// CCTAG_TCOUT( "imax : " << x_sol(4) ); //don't delete.

	// Point raffiné à retourner :
	res.x() = ( x_sol(2) );
	res.y() = ( _a * x_sol(2) + _b );

	//CCTAG_TCOUT( "p0raffine : (" << res ); //don't delete.

	objfcn.cleanup();

	return res;
}

#endif // SUBPIX_EDGE_OPTIM // undefined. Depreciated

} // namespace cctag
