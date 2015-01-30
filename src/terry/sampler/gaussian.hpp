#ifndef _TERRY_SAMPLER_GAUSSIAN_HPP_
#define _TERRY_SAMPLER_GAUSSIAN_HPP_

#include "details.hpp"

namespace terry {
using namespace boost::gil;
namespace sampler {

#ifndef M_PI
/** @brief The constant pi */
#define M_PI    3.14159265358979323846264338327950288
#endif


struct gaussian_sampler
{
	size_t size;
	double sigma;
	gaussian_sampler()
	{
		size  = 3.0;
		sigma = 1.0;
	}
	/*
	void operator()( const float& distance, double& weight, gaussian_sampler& sampler )
	{
		if( sampler.sigma == 0.0 )
		{
			weight = 0.0;
			return;
		}
		weight = 1.0 / ( sampler.sigma * std::sqrt( 2 * boost::math::constant::pi<double>() )) * std::exp( - distance * distance / ( 2 * sampler.sigma ) ) ;
	}*/
};

template < typename F >
void getGaussianWeight( const float& distance, F& weight, gaussian_sampler& sampler )
{
	if( sampler.sigma == 0.0 )
	{
		weight = 0.0;
		return;
	}
	weight = 1.0 / ( sampler.sigma * std::sqrt( 2 * M_PI )) * std::exp( - distance * distance / ( 2 * sampler.sigma ) ) ;
}

template <typename DstP, typename SrcView, typename F>
bool sample( gaussian_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	/*
	 * pTL is the closest integer coordinate top left from p
	 *
	 *   pTL ---> x      x
	 *              o <------ p
	 *
	 *            x      x
	 */
	point2<std::ptrdiff_t> pTL( ifloor( p ) ); //

	// loc is the point in the source view
	typedef typename SrcView::xy_locator xy_locator;
	xy_locator loc = src.xy_at( pTL.x, pTL.y );
	point2<F> frac( p.x - pTL.x, p.y - pTL.y );

	size_t windowSize  = sampler.size;

	std::vector<double> xWeights, yWeights;

	xWeights.assign( windowSize , 0);
	yWeights.assign( windowSize , 0);

	size_t middlePosition = floor((windowSize - 1) * 0.5);


	// get horizontal weight for each pixels

	for( size_t i = 0; i < windowSize; i++ )
	{
		float distancex = - frac.x - middlePosition + i ;
		// sampler( std::abs( distancex ), xWeights.at(i), sampler );
		getGaussianWeight( std::abs( distancex ), xWeights.at(i), sampler );
		float distancey =  - frac.y - middlePosition + i ;
		getGaussianWeight( std::abs( distancey ), yWeights.at(i), sampler );
	}

	// process current sample
	bool res = details::process2Dresampling( sampler, src, p, xWeights, yWeights, windowSize, outOfImageProcess, loc, result );

	return res;
}


}
}

#endif

