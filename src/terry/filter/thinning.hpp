#ifndef _TERRY_FILTER_THINNING_HPP_
#define	_TERRY_FILTER_THINNING_HPP_

#include <terry/channel.hpp>
#include <terry/math/Rect.hpp>
#include <terry/algorithm/transform_pixels.hpp>
#include <terry/numeric/operations.hpp>
#include <terry/numeric/assign_minmax.hpp>
#include <terry/numeric/init.hpp>

namespace terry {
namespace filter {
namespace thinning {

static const bool lutthin1[512] = { false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, false, true, true, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, true, true, false, false, true, true, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, true, false, false, true, true, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, true, false, false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true };
static const bool lutthin2[512] = { false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, false, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, true, true, true, false, false, true, true, false, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, true, true, true, false, false, true, true, false, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, true, true, true, false, false, true, true, false, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, true, true, true, false, false, true, true, false, true, true, true };

template<class SView, class DView=SView>
struct pixel_locator_thinning_t
{
	typedef typename SView::locator SLocator;
	typedef typename DView::locator DLocator;
	typedef typename SView::value_type SPixel;
	typedef typename DView::value_type DPixel;
	typedef typename SLocator::cached_location_t SCachedLocation;

	const bool* _lut;
	SPixel _sWhite;
	DPixel _dWhite;
	DPixel _dBlack;
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

	pixel_locator_thinning_t( const SView& src, const bool* lut )
	: _lut(lut)
	, _loc_ref(src.xy_at(0,0))
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
		pixel_assigns_max( _sWhite );
		pixel_assigns_min( _dBlack );
		pixel_assigns_max( _dWhite );
	}

	DPixel operator()( const SLocator& src ) const
	{
		using namespace terry;

		if( *src != _sWhite )
		{
			return _dBlack;
		}
		
		std::size_t id =  ( src[LT] == _sWhite )       |
			             (( src[LC] == _sWhite ) << 1) |
			             (( src[LB] == _sWhite ) << 2) |
			             (( src[CT] == _sWhite ) << 3) |
			             (( *src    == _sWhite ) << 4) |
			             (( src[CB] == _sWhite ) << 5) |
			             (( src[RT] == _sWhite ) << 6) |
			             (( src[RC] == _sWhite ) << 7) |
			             (( src[RB] == _sWhite ) << 8);
		
		if( _lut[id] )
		{
			return _dWhite;
		}
		return _dBlack;
	}
};



}


template<class SView, class DView>
void applyThinning( const SView& srcView, DView& tmpView, DView& dstView )
{
	using namespace terry::numeric;
	
	typedef typename DView::value_type DPixel;
	DPixel pixelZero; pixel_zeros_t<DPixel>()( pixelZero );
	
	// todo: only fill borders !!
	fill_pixels( tmpView, pixelZero );

	const Rect<std::ptrdiff_t> srcRod = getBounds<std::ptrdiff_t>(srcView);
	const Rect<std::ptrdiff_t> proc1 = rectangleReduce( srcRod, 1 );
	const Rect<std::ptrdiff_t> proc2 = rectangleReduce( proc1, 1 );

	algorithm::transform_pixels_locator(
		srcView, srcRod,
		tmpView, getBounds<std::ptrdiff_t>(tmpView),
		proc1,
		terry::filter::thinning::pixel_locator_thinning_t<SView,DView>(srcView, terry::filter::thinning::lutthin1)
		);

	// todo: only fill borders !!
	fill_pixels( dstView, pixelZero );
	
	algorithm::transform_pixels_locator(
		tmpView, getBounds<std::ptrdiff_t>(tmpView),
		dstView, getBounds<std::ptrdiff_t>(dstView),
		proc2,
		terry::filter::thinning::pixel_locator_thinning_t<DView,DView>(tmpView, terry::filter::thinning::lutthin2)
		);
}

}
}

#endif

