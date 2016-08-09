#ifdef USE_IMAGE_CENTER_OPT // undefined. Depreciated

#include <cctag/optimization/ImageCenterOptimizer.hpp>
#include <cctag/Identification.hpp>
#include <cctag/utils/VisualDebug.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/algebra/Invert.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/utils/Exceptions.hpp>
#include <cctag/utils/Defines.hpp>

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

  x(1) = _pToRefine.x();
  x(2) = _pToRefine.y();
}

void ImageCenterOptimizer::optimizePointFun( int n, const NEWMAT::ColumnVector& x, float& fx, int& result, void *objPtr )
{
	using namespace OPTPP;

	if( n != 2 )
	{
		return;
	}

	This *this_ptr = static_cast<This*>( objPtr );

	cctag::Point2d<Eigen::Vector3f> centerExtEllipse( x(1), x(2) );

	cctag::numerical::optimization::condition(centerExtEllipse, this_ptr->_mInvT);

	CCTagVisualDebug::instance().drawPoint( centerExtEllipse, cctag::color_blue );

	Eigen::Matrix3f mH;
	VecSignals vecSig;
	if ( !getSignals( mH, vecSig, this_ptr->_lengthSig, centerExtEllipse, this_ptr->_vecExtPoints, this_ptr->_src, this_ptr->_ellipse.matrix() ) )
	{
		// We are diverging
		CCTAG_COUT_DEBUG("divergence!");
		return;
	}

	float res = 0;
        std::size_t resSize = 0;
	for( std::size_t i = 0; i < vecSig.size() - 1; ++i )
	{
		for( std::size_t j = i+1; j < vecSig.size(); ++j )
		{
			res += std::pow( norm_2( vecSig[i]._imgSignal - vecSig[j]._imgSignal ), 4 );
                        ++resSize;
		}
	}
        res /= resSize;

	++this_ptr->_numIter;

	fx = res;
	result = NLPFunction;
}

Point2d<Eigen::Vector3f> ImageCenterOptimizer::operator()( const cctag::Point2d<Eigen::Vector3f> & pToRefine, 
                const std::size_t lengthSig,
                const cv::Mat & src,
                const cctag::numerical::geometry::Ellipse & outerEllipse,
                const Eigen::Matrix3f & mT)
{
	using namespace OPTPP;
	using namespace NEWMAT;

	Point2d<Eigen::Vector3f> res;

	_pToRefine = pToRefine;
	cctag::numerical::optimization::condition(_pToRefine, mT);
	
	_lengthSig = lengthSig;
	_src = src;
	_ellipse = outerEllipse;
	_numIter = 0;
	_mT = mT;
	
	cctag::numerical::invert_3x3(mT,_mInvT);

	OptQNewton objfcn( this );

	objfcn.setSearchStrategy( TrustRegion );
	objfcn.setMaxFeval( 200 );
	objfcn.setFcnTol( 1.0e-4 );
#if defined(DEBUG)
	if ( !objfcn.setOutputFile("example1.out", 0) )
	{
		CCTAG_COUT_ERROR( "main: output file open failed" );
	}
#endif

	objfcn.optimize();

	objfcn.cleanup();

	//#ifdef REG_TEST
	ColumnVector x_sol = getXc();
	// Point raffiné à retourner :
	res.x() = x_sol( 1 );
	res.y() = x_sol( 2 );

	cctag::numerical::optimization::condition(res, _mInvT);

	objfcn.cleanup();

	return res;
}

} // namespace identification
} // namespace cctag

#endif // USE_IMAGE_CENTER_OPT
