#ifndef _TERRY_FILTER_LOCALMAXIMA_HPP_
#define _TERRY_FILTER_LOCALMAXIMA_HPP_

#include <terry/channel.hpp>
#include <terry/math/Rect.hpp>
#include <terry/algorithm/transform_pixels.hpp>
#include <terry/numeric/operations.hpp>
#include <terry/numeric/init.hpp>
#include <terry/numeric/assign_minmax.hpp>

#include <boost/assert.hpp>


namespace terry {
namespace filter {


/**
 * Computation of gradient norm local maxima in regard of gradient direction
 *
 *     there are 4 cases:
 *
 *                          The X marks the pixel in question, and each
 *          C     B         of the quadrants for the gradient vector
 *        O----0----0       fall into two cases, divided by the 45
 *      D |         | A     degree line.  In one case the gradient
 *        |         |       vector is more horizontal, and in the other
 *        O    X    O       it is more vertical.  There are eight
 *        |         |       divisions, but for the non-maximum suppression
 *     (A)|         |(D)    we are only worried about 4 of them since we
 *        O----O----O       use symmetric points about the center pixel.
 *         (B)   (C)
 */
template<class SView, class DView=SView>
struct pixel_locator_gradientLocalMaxima_t
{
	typedef typename SView::locator SLocator;
	typedef typename SView::value_type SPixel;
	typedef typename boost::gil::channel_type<SPixel>::type SChannel;
	typedef typename terry::channel_base_type<SChannel>::type SType;
	typedef typename SLocator::cached_location_t SCachedLocation;

	typedef typename DView::locator DLocator;
	typedef typename DView::value_type DPixel;

	DPixel _black;
	const SLocator _loc_ref;
	// LT CT RT
	// LC    RC
	// LB CB RB
	const SCachedLocation LT;
	const SCachedLocation CT;
	const SCachedLocation RT;
	const SCachedLocation LC;
	const SCachedLocation RC;
	const SCachedLocation LB;
	const SCachedLocation CB;
	const SCachedLocation RB;

	static const unsigned int vecX = 0;
	static const unsigned int vecY = 1;
	static const unsigned int norm = 2;

	pixel_locator_gradientLocalMaxima_t( const SView& src )
	: _loc_ref(src.xy_at(0,0))
	, LT(_loc_ref.cache_location(-1,-1))
	, CT(_loc_ref.cache_location( 0,-1))
	, RT(_loc_ref.cache_location( 1,-1))

	, LC(_loc_ref.cache_location(-1, 0))
	, RC(_loc_ref.cache_location( 1, 0))

	, LB(_loc_ref.cache_location(-1, 1))
	, CB(_loc_ref.cache_location( 0, 1))
	, RB(_loc_ref.cache_location( 1, 1))
	{
		using namespace terry::numeric;
		pixel_assigns_min( _black );
	}

	DPixel operator()( const SLocator& src ) const
	{
		using namespace terry;
		
		SType g1;
		SType g2;

		// A
		if( ((*src)[vecY] <= 0 && (*src)[vecX] > -(*src)[vecY]) ||
			((*src)[vecY] >= 0 && (*src)[vecX] < -(*src)[vecY]) )
		{
			SType d = 0.0;
			if( (*src)[vecX] )
			{
				d = std::abs( (*src)[vecY] / (*src)[vecX] );
			}
			SType invd = 1.0 - d;
			// __________
			// |__|__|RT|
			// |LC|__|RC|
			// |LB|__|__|
			g1 = src[RC][norm] * invd + src[RT][norm] * d;
			g2 = src[LC][norm] * invd + src[LB][norm] * d;
		}
		// B
		else if ( ((*src)[vecX] > 0 && -(*src)[vecY] >= (*src)[vecX]) ||
				  ((*src)[vecX] < 0 && -(*src)[vecY] <= (*src)[vecX]) )
		{
			SType d = 0.0;
			if( (*src)[vecY] )
			{
				d = std::abs( (*src)[vecX] / (*src)[vecY] );
			}
			SType invd = 1.0 - d;
			// __________
			// |__|CT|RT|
			// |__|__|__|
			// |LB|CB|__|
			g1 = src[CT][norm] * invd + src[RT][norm] * d;
			g2 = src[CB][norm] * invd + src[LB][norm] * d;
		}
		// C
		else if( ((*src)[vecX] <= 0 && (*src)[vecX] > (*src)[vecY]) ||
				 ((*src)[vecX] >= 0 && (*src)[vecX] < (*src)[vecY]) )
		{
			SType d = 0.0;
			if( (*src)[vecY] )
			{
				d = std::abs( (*src)[vecX] / (*src)[vecY] );
			}
			SType invd = 1.0 - d;
			// __________
			// |LT|CT|__|
			// |__|__|__|
			// |__|CB|RB|
			g1 = src[CT][norm] * invd + src[LT][norm] * d;
			g2 = src[CB][norm] * invd + src[RB][norm] * d;
		}
		// D
//		else if( ((*src)[vecY] < 0 && (*src)[vecX] <= (*src)[vecY]) ||
//		         ((*src)[vecY] > 0 && (*src)[vecX] >= (*src)[vecY]) )
		else
		{
			BOOST_ASSERT( ((*src)[vecY] < 0 && (*src)[vecX] <= (*src)[vecY]) ||
			              ((*src)[vecY] > 0 && (*src)[vecX] >= (*src)[vecY]) );
			SType d = 0.0;
			if( (*src)[vecX] )
			{
				d = std::abs( (*src)[vecY] / (*src)[vecX] );
			}
			SType invd = 1.0 - d;
			// __________
			// |LT|__|__|
			// |LC|__|RC|
			// |__|__|RB|
			g1 = src[LC][norm] * invd + src[LT][norm] * d;
			g2 = src[RC][norm] * invd + src[RB][norm] * d;
		}
		DPixel dst = _black;
		if( (*src)[norm] >= g1 && (*src)[norm] >= g2 )
		{
			static_fill( dst, (*src)[norm] ); // wins !
		}
		return dst;
	}
};


template<class SView, class DView>
void applyLocalMaxima( const SView& srcView, DView& dstView )
{
	using namespace terry::numeric;
	
	typedef typename DView::value_type DPixel;
	DPixel pixelZero; pixel_zeros_t<DPixel>()( pixelZero );
	
	// todo: only fill borders !!
	fill_pixels( dstView, pixelZero );

	terry::algorithm::transform_pixels_locator(
		srcView, getBounds<std::ptrdiff_t>(srcView),
		dstView, getBounds<std::ptrdiff_t>(dstView),
		getBounds<std::ptrdiff_t>(dstView),
		terry::filter::pixel_locator_gradientLocalMaxima_t<SView,DView>(srcView)
		);
}




}
}

#endif
