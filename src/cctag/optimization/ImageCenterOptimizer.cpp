#ifdef USE_IMAGE_CENTER_OPT // undefined. Depreciated

#include <cctag/optimization/ImageCenterOptimizer.hpp>
#include <cctag/Identification.hpp>
#include <cctag/VisualDebug.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/algebra/Invert.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/utils/exceptions.hpp>
#include <cctag/Global.hpp>

#include <OptQNewton.h>

#include <terry/sampler/all.hpp>

#include <boost/bind.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <ostream>

namespace cctag {
namespace identification {

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

  x(1) = _pToRefine.x();// todo@Lilian: why not (0) and (1) instead ?!
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

	cctag::Point2dN<double> centerExtEllipse( x(1), x(2) );

	cctag::numerical::optimization::condition(centerExtEllipse, this_ptr->_mInvT);

	//CCTAG_TCOUT_VAR( centerExtEllipse );

	//CCTagVisualDebug::instance().drawText( centerExtEllipse, boost::lexical_cast<std::string>(this_ptr->_numIter), cctag::color_white );
	CCTagVisualDebug::instance().drawPoint( centerExtEllipse, cctag::color_blue );

	cctag::numerical::BoundedMatrix3x3d mH;
	VecSignals vecSig;
	if ( !getSignals( mH, vecSig, this_ptr->_lengthSig, centerExtEllipse, this_ptr->_vecExtPoints, this_ptr->_src, this_ptr->_ellipse.matrix() ) )
	{
		// We are diverging
		CCTAG_COUT_DEBUG("divergence!");
		return;
	}

	double res = 0;
        std::size_t resSize = 0;
	for( std::size_t i = 0; i < vecSig.size() - 1; ++i )
	{
		//CCTAG_TCOUT_VAR(vecSig[i]._imgSignal);
		for( std::size_t j = i+1; j < vecSig.size(); ++j )
		{
			res += std::pow( norm_2( vecSig[i]._imgSignal - vecSig[j]._imgSignal ), 4 );
                        //res += norm_2( vecSig[i]._imgSignal - vecSig[j]._imgSignal );
                        ++resSize;
		}
	}
        res /= resSize;

	++this_ptr->_numIter;

	//double penalty = 0;
	//double distanceToCentre = cctag::numerical::distancePoints2D( centerExtEllipse, this_ptr->_ellipse.center() );
	//if ( distanceToCentre > 0.2*std::min(this_ptr->_ellipse.a(), this_ptr->_ellipse.b()) )
	//{
	//	penalty += 1000000*distanceToCentre/std::min(this_ptr->_ellipse.a(), this_ptr->_ellipse.b());
	//}

	fx = res;//+penalty;
	result = NLPFunction;
}

Point2dN<double> ImageCenterOptimizer::operator()( const cctag::Point2dN<double> & pToRefine, const std::size_t lengthSig, const cv::Mat & src, const cctag::numerical::geometry::Ellipse & outerEllipse, const cctag::numerical::BoundedMatrix3x3d & mT)
{
	using namespace OPTPP;
	using namespace NEWMAT;

	Point2dN<double> res;

	//  Create a Nonlinear problem object
	_pToRefine = pToRefine;
	cctag::numerical::optimization::condition(_pToRefine, mT);
	
	_lengthSig = lengthSig;
	_src = src;
	_ellipse = outerEllipse;
	_numIter = 0;
	// 2D conditioning matrix
	_mT = mT;
	
	cctag::numerical::invert_3x3(mT,_mInvT);

	OptQNewton objfcn( this );

	objfcn.setSearchStrategy( TrustRegion );//LineSearch ); //TrustRegion );
	objfcn.setMaxFeval( 200 );
	objfcn.setFcnTol( 1.0e-4 );
    //objfcn.setMaxStep(0.2);
#if defined(DEBUG)
	if ( !objfcn.setOutputFile("example1.out", 0) )
	{
		CCTAG_COUT_ERROR( "main: output file open failed" );
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

	cctag::numerical::optimization::condition(res, _mInvT);

	objfcn.cleanup();

	return res;
}

} // namespace identification
} // namespace cctag

#endif // USE_IMAGE_CENTER_OPT
