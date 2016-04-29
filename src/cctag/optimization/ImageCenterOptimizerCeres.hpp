#ifdef USE_IMAGE_CENTER_OPT_CERES // undefined. Depreciated

#ifndef VISION_IMAGECENTEROPTIMIZERCERES_HPP
#define	VISION_IMAGECENTEROPTIMIZERCERES_HPP

#include <cctag/utils/VisualDebug.hpp>
#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/algebra/Invert.hpp>
#include <cctag/optimization/conditioner.hpp>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "Identification.hpp"

#include <cstddef>
#include <vector>

namespace cctag {
namespace identification {

struct TotoFunctor {

	typedef std::vector< cctag::Point2d<Eigen::Vector3f> > VecExtPoints;
	typedef std::vector< cctag::ImageCut > VecSignals;


  TotoFunctor( const VecExtPoints & vecExtPoints, const std::size_t lengthSig, const cv::Mat & src,
  const cctag::numerical::geometry::Ellipse & outerEllipse, const Eigen::Matrix3f & mT)
      : _vecExtPoints(vecExtPoints), _lengthSig(lengthSig), _src(src), _ellipse(outerEllipse), _mT(mT) {

	  cctag::numerical::invert_3x3(mT,_mInvT);
  }


    bool operator()(const float* const x, float* residual) const {
	
    cctag::Point2d<Eigen::Vector3f> centerExtEllipse( x[0], x[1] );

	cctag::numerical::optimization::condition(centerExtEllipse, _mInvT);
	//CCTAG_TCOUT_VAR( centerExtEllipse );
	//CCTagVisualDebug::instance().drawText( centerExtEllipse, boost::lexical_cast<std::string>(this_ptr->_numIter), cctag::color_white );
	CCTagVisualDebug::instance().drawPoint( centerExtEllipse, cctag::color_blue );

	Eigen::Matrix3f mH;
	VecSignals vecSig;
	if ( !getSignals( mH, vecSig, _lengthSig, centerExtEllipse, _vecExtPoints, _src, _ellipse.matrix() ) )
	{
		// We are diverging
		CCTAG_COUT_DEBUG("divergence!");
		return false;
	}

	residual[0] = 0;

	for( std::size_t i = 0; i < vecSig.size() - 1; ++i )
	{
		//CCTAG_TCOUT_VAR(vecSig[i]._imgSignal);
		for( std::size_t j = i+1; j < vecSig.size(); ++j )
		{
			residual[0] += std::pow( norm_2( vecSig[i]._imgSignal - vecSig[j]._imgSignal ), 2 );
		}
	}
    return true;
  }


private:
	const VecExtPoints & _vecExtPoints;
	std::size_t _lengthSig;
	cv::Mat _src;
	cctag::numerical::geometry::Ellipse _ellipse;
	Eigen::Matrix3f _mT;
	Eigen::Matrix3f _mInvT;

};

void optimizeCenterCeres();

} // namespace identification
} // namespace cctag



#endif	/* VISION_IMAGECENTEROPTIMIZERCERES_HPP */

#endif // undefined. Depreciated
