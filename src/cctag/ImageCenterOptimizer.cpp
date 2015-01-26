#include "modeConfig.hpp"

#include "ImageCenterOptimizer.hpp"
#include "identification.hpp"
#include "visualDebug.hpp"

#include <cctag/geometry/point.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/progBase/exceptions.hpp>
#include <cctag/global.hpp>

#include "OptQNewton.h"

#include <terry/sampler/all.hpp>

#include <boost/bind.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <ostream>

namespace rom {
namespace vision {
namespace marker {

ImageCenterOptimizer::ImageCenterOptimizer( const VecExtPoints & vecExtPoints )
: Parent( 2, &ImageCenterOptimizer::optimizePointFun, NULL, this )
, _vecExtPoints( vecExtPoints )
{
}

void ImageCenterOptimizer::initOpt( int ndim, NEWMAT::ColumnVector& x )
{
	if ( ndim != 2 )
	{
		BOOST_THROW_EXCEPTION( exception::Bug() << exception::dev() + "Unable to init minimizer!" );
	}

	x(1) = _pToRefine.x();
	x(2) = _pToRefine.y();
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
void ImageCenterOptimizer::optimizePointFun( int n, const NEWMAT::ColumnVector& x, double& fx, int& result, void *objPtr )
{
	using namespace OPTPP;

	if( n != 2 )
	{
		return;
	}

	This *this_ptr = static_cast<This*>( objPtr );

	rom::Point2dN<double> centerExtEllipse( x(1), x(2) );

	rom::numerical::optimization::condition(centerExtEllipse, this_ptr->_mInvT);

	//ROM_TCOUT_VAR( centerExtEllipse );

	//CCTagVisualDebug::instance().drawText( centerExtEllipse, boost::lexical_cast<std::string>(this_ptr->_numIter), rom::color_white );
	CCTagVisualDebug::instance().drawPoint( centerExtEllipse, rom::color_blue );

	rom::numerical::BoundedMatrix3x3d mH;
	VecSignals vecSig;
	if ( !getSignals( mH, vecSig, this_ptr->_lengthSig, centerExtEllipse, this_ptr->_vecExtPoints, this_ptr->_sView, this_ptr->_ellipse.matrix() ) )
	{
		// We are diverging
		ROM_COUT_DEBUG("divergence!");
		return;
	}

	double res = 0;

	for( std::size_t i = 0; i < vecSig.size() - 1; ++i )
	{
		//ROM_TCOUT_VAR(vecSig[i]._imgSignal);
		for( std::size_t j = i+1; j < vecSig.size(); ++j )
		{
			res += std::pow( norm_2( vecSig[i]._imgSignal - vecSig[j]._imgSignal ), 2 );
		}
	}

	++this_ptr->_numIter;

	//double penalty = 0;
	//double distanceToCentre = rom::numerical::distancePoints2D( centerExtEllipse, this_ptr->_ellipse.center() );
	//if ( distanceToCentre > 0.2*std::min(this_ptr->_ellipse.a(), this_ptr->_ellipse.b()) )
	//{
	//	penalty += 1000000*distanceToCentre/std::min(this_ptr->_ellipse.a(), this_ptr->_ellipse.b());
	//}

	fx = res;//+penalty;
	result = NLPFunction;
}

Point2dN<double> ImageCenterOptimizer::operator()( const rom::Point2dN<double> & pToRefine, const std::size_t lengthSig, const boost::gil::gray8_view_t & sView, const rom::numerical::geometry::Ellipse & outerEllipse, const rom::numerical::BoundedMatrix3x3d & mT)
{
	using namespace OPTPP;
	using namespace NEWMAT;

	Point2dN<double> res;

	//  Create a Nonlinear problem object
	_pToRefine = pToRefine;
	rom::numerical::optimization::condition(_pToRefine, mT);
	
	_lengthSig = lengthSig;
	_sView = sView;
	_ellipse = outerEllipse;
	_numIter = 0;
	// 2D conditioning matrix
	_mT = mT;
	
	rom::numerical::invert_3x3(mT,_mInvT);

	OptQNewton objfcn( this );

	objfcn.setSearchStrategy( LineSearch ); //TrustRegion );
	objfcn.setMaxFeval( 200 );
	objfcn.setFcnTol( 1.0e-4 );

#if defined(DEBUG) || defined(CCTAG_STAT_DEBUG)
	if ( !objfcn.setOutputFile("example1.out", 0) )
	{
		ROM_COUT_ERROR( "main: output file open failed" );
	}
#endif

	objfcn.optimize();

	//objfcn.printStatus( "Solution from quasi-newton" );
	objfcn.cleanup();

	//#ifdef REG_TEST
	ColumnVector x_sol = getXc();
	// Point raffiné à retourner :
	res.setX( x_sol( 1 ) );
	res.setY( x_sol( 2 ) );

	rom::numerical::optimization::condition(res, _mInvT);

	objfcn.cleanup();

	return res;
}

}
}
}
