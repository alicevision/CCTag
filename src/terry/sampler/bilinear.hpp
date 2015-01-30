#ifndef _TERRY_SAMPLER_BILINEAR_HPP_
#define _TERRY_SAMPLER_BILINEAR_HPP_

#include "details.hpp"

namespace terry {
using namespace boost::gil;
namespace sampler {


struct bilinear_sampler{};

/**
 * @brief Get weight for a specific distance, for all bilinear resampler.
 *
 * @param[in] distance between the pixels and the current pixel
 * @param[out] weight return value to weight the pixel in filtering
 */
template< typename F >
bool getWeight( const long int&  pTLXOrY, const F& distance, const size_t index, F& weight )
{
	if( distance < 1 )
	{
		weight = (1.0 - distance);
		return true;
	}
	else
	{
		if( distance < 2 )
		{
			weight = 0;
			return true;
		}
		return false;
	}
}

template <typename DstP, typename SrcView, typename F>
bool sample( bilinear_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
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

		int windowSize  = 2;

		std::vector<double> xWeights, yWeights;

		xWeights.assign( windowSize , 0);
		yWeights.assign( windowSize , 0);

		// get weight for each pixels
		for(int i=0; i<windowSize; i++)
		{
			getWeight( pTL.x, std::abs( (double)(i - frac.x) ), i, xWeights.at(i) );
			getWeight( pTL.y, std::abs( (double)(i - frac.y) ), i, yWeights.at(i) );
		}

		// process current sample
		return details::process2Dresampling( sampler, src, p, xWeights, yWeights, windowSize, outOfImageProcess, loc, result );
}



}
}

#endif

