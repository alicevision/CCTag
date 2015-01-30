#ifndef _TERRY_SAMPLER_BC_HPP_
#define _TERRY_SAMPLER_BC_HPP_

#include "details.hpp"

namespace terry {
using namespace boost::gil;
namespace sampler {

struct bc_sampler
{
	double valB;
	double valC;
};

//
// valC is equal to -a in the equation
//
struct cubic_sampler : bc_sampler
{
	cubic_sampler()
	{
		valB = 0.0;
	}
	void setAValueTo( double a ){valC = -a;}
};

struct bicubic_sampler : cubic_sampler
{
	bicubic_sampler()
	{
		cubic_sampler::setAValueTo( 0.0 );
	}
};

// catmul-rom resampling function
struct catrom_sampler : cubic_sampler
{
	catrom_sampler()
	{
		cubic_sampler::setAValueTo( -0.5 );
	}
};

// similar to catrom resampling function
struct keys_sampler : cubic_sampler
{
	keys_sampler()
	{
		cubic_sampler::setAValueTo( -0.5 );
	}
};

// similar to catrom resampling function
struct simon_sampler : cubic_sampler
{
	simon_sampler()
	{
		cubic_sampler::setAValueTo( -0.75 );
	}
};

// similar to catrom resampling function
struct rifman_sampler : cubic_sampler
{
	rifman_sampler()
	{
		cubic_sampler::setAValueTo( -1.0 );
	}
};

struct mitchell_sampler : bc_sampler
{
	mitchell_sampler()
	{
		valB = 1.0/3.0;
		valC = 1.0/3.0;
	}
};

struct parzen_sampler : bc_sampler
{
	parzen_sampler()
	{
		valB = 1.0;
		valC = 0.0;
	}
};

/**
 * @brief Get weight for a specific distance, for all BC-cubic resampling (bicubic, catmul-rom, ...).
 *
 * For compute cubic BC resampler weights, we use these functions
 * [ Reconstruction Filters in Computer Graphics,
 *   Don P. Mitchell, Arun N. Netravali,
 *   Computer Graphics - volume 22 number 4 - August 1988
 *   <a href="http://www.cs.utexas.edu/users/fussell/courses/cs384g/lectures/mitchell/Mitchell.pdf">online paper</a>
 * ]:
 *
 * \f[ W(x) =
 * \begin{cases}
 * (a+2)|x|^3-(a+3)|x|^2+1 & \text{for } |x| \leq 1 \\
 * a|x|^3-5a|x|^2+8a|x|-4a & \text{for } 1 < |x| < 2 \\
 * 0                       & \text{otherwise}
 * \end{cases}
 * \f]
 * @param[in] B value of B in BC-cubic resampling function
 * @param[in] C value of C in BC-cubic resampling function
 * @param[in] distance between the pixels and the current pixel
 * @param[out] weight return value to weight the pixel in filtering
**/
template < typename F >
bool getWeight ( const size_t& index, const double& distance, F& weight, bc_sampler& sampler )
{
	if( distance <= 1 )
	{
		double P =  12.0 -  9.0 * sampler.valB - 6.0 * sampler.valC;
		double Q = -18.0 + 12.0 * sampler.valB + 6.0 * sampler.valC;
		double S =   6.0 -  2.0 * sampler.valB;
		// note: R is null
		weight = ( ( P * distance + Q ) *  distance * distance + S ) / 6.0;
		return true;
	}
	else
	{
		if( distance < 2 )
		{
			double T = -        sampler.valB -  6.0 * sampler.valC;
			double U =    6.0 * sampler.valB + 30.0 * sampler.valC;
			double V = - 12.0 * sampler.valB - 48.0 * sampler.valC;
			double W =    8.0 * sampler.valB + 24.0 * sampler.valC;
			weight = ( ( ( T * distance + U ) *  distance + V ) * distance + W ) / 6.0;
			return true;
		}
		weight = 0.0;
		return false;
	}
}

template <typename DstP, typename SrcView, typename F>
bool sample( bc_sampler sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
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

	int windowSize  = 4;             // 4 pixels:    A B C D

	std::vector<double> xWeights, yWeights;

	xWeights.assign( windowSize , 0);
	yWeights.assign( windowSize , 0);

	//xWeights.at(1) = 1.0;

	// get horizontal weight for each pixels
	for( int i = 0; i < windowSize; i++ )
	{
		getWeight( i, std::abs( (i-1) - frac.x ), xWeights.at(i), sampler );
		getWeight( i, std::abs( (i-1) - frac.y ), yWeights.at(i), sampler );
	}

	//TUTTLE_COUT ("point " << "weights = " << xWeights.at(0) << "; " << yWeights.at(0) );
	//process2Dresampling( Sampler& sampler, const SrcView& src, const point2<F>& p, const std::vector<double>& xWeights, const std::vector<double>& yWeights, const size_t& windowSize,typename SrcView::xy_locator& loc, DstP& result )
	return details::process2Dresampling( sampler, src, p, xWeights, yWeights, windowSize, outOfImageProcess, loc, result );
}

}
}

#endif
