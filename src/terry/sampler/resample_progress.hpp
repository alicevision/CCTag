#ifndef _TERRY_SAMPLER_RESAMPLE_HPP_
#define	_TERRY_SAMPLER_RESAMPLE_HPP_

#include <terry/math/Rect.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/sampler.hpp>


namespace terry {
namespace sampler {

/**
 * @brief Set each pixel in the destination view as the result of a sampling function over the transformed coordinates of the source view
 * @ingroup ImageAlgorithms
 *
 * The provided implementation works for 2D image views only
 */
template<
	typename Sampler, // Models SamplerConcept
	typename SrcView, // Models RandomAccess2DImageViewConcept
	typename DstView, // Models MutableRandomAccess2DImageViewConcept
	typename MapFn,
	typename Progress>
// Models MappingFunctionConcept
void resample_pixels_progress(
	const SrcView& src_view, const DstView& dst_view,
	const MapFn& dst_to_src, const terry::Rect<std::ssize_t>& procWindow,
	const EParamFilterOutOfImage& outOfImageProcess,
	Progress& p,
	Sampler sampler = Sampler() )
{
	typedef typename DstView::point_t Point2;
	typedef typename DstView::value_type Pixel;
	Point2 dst_p;
	Pixel black;
	color_convert( boost::gil::rgba32f_pixel_t( 0.0, 0.0, 0.0, 0.0 ), black );
	for( dst_p.y = procWindow.y1; dst_p.y < procWindow.y2; ++dst_p.y )
	{
		typename DstView::x_iterator xit = dst_view.row_begin( dst_p.y );
		for( dst_p.x = procWindow.x1; dst_p.x < procWindow.x2; ++dst_p.x )
		{

			if( ! sample( sampler, src_view, transform( dst_to_src, dst_p ), xit[dst_p.x], outOfImageProcess ) )
			{
				xit[dst_p.x] = black; // if it is outside of the source image
			}
		}
		if( p.progressForward() )
			return;
	}
}

}
}

#endif

