/* 
 * File:   ImageCenterOptimizerCeres.hpp
 * Author: lcalvet
 *
 * Created on 15 mai 2014, 14:52
 */

#ifndef _POPART_VISION_IMAGECENTEROPTIMIZERCERES_HPP
#define	_POPART_VISION_IMAGECENTEROPTIMIZERCERES_HPP

#include "visualDebug.hpp"

#include <cctag/global.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/imageCut.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/optimization/conditioner.hpp>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "identification.hpp"

#include <cstddef>
#include <vector>

namespace popart {
namespace vision {
namespace marker {

struct TotoFunctor {

	typedef std::vector< popart::Point2dN<double> > VecExtPoints;
	typedef std::vector< popart::ImageCut > VecSignals;


  TotoFunctor( const VecExtPoints & vecExtPoints, const std::size_t lengthSig, const boost::gil::gray8_view_t & sView,
  const popart::numerical::geometry::Ellipse & outerEllipse, const popart::numerical::BoundedMatrix3x3d & mT)
      : _vecExtPoints(vecExtPoints), _lengthSig(lengthSig), _sView(sView), _ellipse(outerEllipse), _mT(mT) {

	  popart::numerical::invert_3x3(mT,_mInvT);
  }


    bool operator()(const double* const x, double* residual) const {
	
    popart::Point2dN<double> centerExtEllipse( x[0], x[1] );

	popart::numerical::optimization::condition(centerExtEllipse, _mInvT);
	//ROM_TCOUT_VAR( centerExtEllipse );
	//CCTagVisualDebug::instance().drawText( centerExtEllipse, boost::lexical_cast<std::string>(this_ptr->_numIter), popart::color_white );
	CCTagVisualDebug::instance().drawPoint( centerExtEllipse, popart::color_blue );

	popart::numerical::BoundedMatrix3x3d mH;
	VecSignals vecSig;
	if ( !getSignals( mH, vecSig, _lengthSig, centerExtEllipse, _vecExtPoints, _sView, _ellipse.matrix() ) )
	{
		// We are diverging
		ROM_COUT_DEBUG("divergence!");
		return false;
	}

	residual[0] = 0;

	for( std::size_t i = 0; i < vecSig.size() - 1; ++i )
	{
		//ROM_TCOUT_VAR(vecSig[i]._imgSignal);
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
	boost::gil::gray8_view_t _sView;
	//popart::numerical::BoundedMatrix3x3d _matEllipse;
	popart::numerical::geometry::Ellipse _ellipse;
	//std::size_t _numIter;
	popart::numerical::BoundedMatrix3x3d _mT;
	popart::numerical::BoundedMatrix3x3d _mInvT;

};

void optimizeCenterCeres();

}}}



#endif	/* _POPART_VISION_IMAGECENTEROPTIMIZERCERES_HPP */

