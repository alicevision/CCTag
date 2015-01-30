#ifndef _TERRY_SAMPLER_LANCZOS_HPP_
#define _TERRY_SAMPLER_LANCZOS_HPP_

#include "details.hpp"
#include <cmath>

#ifndef M_PI
/** @brief The constant pi */
#define M_PI    3.14159265358979323846264338327950288
#endif

namespace terry {
using namespace boost::gil;
namespace sampler {

struct lanczos_sampler{
	size_t size;
	lanczos_sampler()
	{
		size = 3.0;
	}
};

struct lanczos3_sampler{};
struct lanczos4_sampler{};
struct lanczos6_sampler{};
struct lanczos12_sampler{};

template < typename F >
void getLanczosWeight( const float& distance, F& weight, lanczos_sampler& sampler )
{
	if( distance == 0.0 )
	{
		weight = 1.0;
		return;
	}
	weight = sin( M_PI * distance ) * sin( ( M_PI / sampler.size ) * distance ) / ( ( M_PI * M_PI / sampler.size ) * distance *distance);
}


template <typename DstP, typename SrcView, typename F>
bool sample( lanczos_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
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
		getLanczosWeight( std::abs( distancex ), xWeights.at(i), sampler );
		float distancey =  - frac.y - middlePosition + i ;
		getLanczosWeight( std::abs( distancey ), yWeights.at(i), sampler );
	}

	// process current sample
	bool res = details::process2Dresampling( sampler, src, p, xWeights, yWeights, windowSize, outOfImageProcess, loc, result );

	return res;
}

template <typename DstP, typename SrcView, typename F>
bool sample( lanczos3_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	lanczos_sampler s;
	s.size = 3;
        return sample( s, src, p, result, outOfImageProcess );
}

template <typename DstP, typename SrcView, typename F>
bool sample( lanczos4_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	lanczos_sampler s;
	s.size = 4;
        return sample( s, src, p, result, outOfImageProcess );
}

template <typename DstP, typename SrcView, typename F>
bool sample( lanczos6_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	lanczos_sampler s;
	s.size = 6;
        return sample( s, src, p, result, outOfImageProcess );
}

template <typename DstP, typename SrcView, typename F>
bool sample( lanczos12_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	lanczos_sampler s;
	s.size = 12;
        return sample( s, src, p, result, outOfImageProcess );
}

}
}

#endif

