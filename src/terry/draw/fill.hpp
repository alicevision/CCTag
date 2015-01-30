#ifndef _TERRY_DRAW_FILL_HPP_
#define _TERRY_DRAW_FILL_HPP_

#include <terry/math/Rect.hpp>

#include <boost/gil/gil_config.hpp>
#include <boost/gil/algorithm.hpp>

namespace terry {
namespace draw {

template<class View>
GIL_FORCEINLINE
void fill_pixels( View& dstView, const typename View::value_type& pixelValue )
{
        typedef typename View::value_type Pixel;
        boost::gil::fill_pixels( dstView, pixelValue );
}


/**
 * @brief fill all pixels inside the @p window, with the color @p pixelValue.
 *
 * @param[out] dstView Image view to fill.
 * @param[in] window Rectangle region of the image to fill.
 * @param[in] pixelValue Pixel value used to fill.
 */
template<class View>
GIL_FORCEINLINE
void fill_pixels( View& dstView, const Rect<std::ssize_t>& window,
				  const typename View::value_type& pixelValue )
{
	typedef typename View::value_type Pixel;

	View dst = subimage_view( dstView, window.x1, window.y1,
	                                   window.x2-window.x1, window.y2-window.y1 );
	boost::gil::fill_pixels( dst, pixelValue );
}

/**
 * @brief fill a range of pixels (on the same line or with an 1d_traversable image).
 *
 * @param[out] dstBegin Begining of the region to fill. The content is filled but the original iterator is not modified (not a reference parameter).
 * @param[in] dstEnd End of the region to fill.
 * @param[in] pixelValue Pixel value used to fill.
 */
template<class DIterator, class DPixel>
GIL_FORCEINLINE
void fill_pixels_range( DIterator dstBegin, const DIterator& dstEnd, const DPixel& pixelValue )
{
	do
	{
		*dstBegin = pixelValue;
		++dstBegin;
	}
	while( dstBegin != dstEnd );
}



}
}

#endif

