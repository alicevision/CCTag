#ifndef _TERRY_NUMERIC_SCALE_HPP_
#define _TERRY_NUMERIC_SCALE_HPP_

#include "assign.hpp"

#include <terry/channel.hpp>

#include <boost/gil/gil_config.hpp>
#include <boost/gil/pixel.hpp>
#include <boost/gil/color_base_algorithm.hpp>


namespace terry {
namespace numeric {

/**
 * @brief Rescale pixel values. Usefull to normalize an image.
 */
template<typename Pixel, typename CPixel = Pixel>
struct pixel_scale_t
{
	typedef typename boost::gil::channel_type<Pixel>::type Channel;
	typedef typename terry::channel_base_type<Channel>::type ChannelBaseType;

	const CPixel _ratio;
	CPixel _shift;

	pixel_scale_t( const CPixel& ratio, const CPixel& sCenter, const CPixel& dCenter )
	: _ratio(ratio)
	{
		BOOST_STATIC_ASSERT( boost::is_floating_point<ChannelBaseType>::value );
		// res = ( ( src - sCenter ) * _ratio ) + dCenter;
		// res = (src * _ratio) + (dCenter - (sCenter*ratio))
		// shift = (dCenter - (sCenter*ratio))
		pixel_assigns_t<CPixel,CPixel>()(
			pixel_minus_t<CPixel,CPixel,CPixel>()(
				dCenter,
				pixel_multiplies_t<CPixel,CPixel,CPixel>()(
					sCenter,
					ratio
					)
				),
			_shift );
	}

	pixel_scale_t( const CPixel& ratio, const CPixel& shift )
	: _ratio(ratio)
	, _shift(shift)
	{
		BOOST_STATIC_ASSERT( boost::is_floating_point<ChannelBaseType>::value );
	}

	GIL_FORCEINLINE
	Pixel operator()( const Pixel& src ) const
	{
		Pixel res;
		// res = (src * _ratio) + _shift
		pixel_assigns_t<CPixel,Pixel>()(
			pixel_plus_t<CPixel,CPixel,CPixel>()(
				pixel_multiplies_t<Pixel,CPixel,CPixel>()(
					src,
					_ratio
					),
				_shift ),
			res );

		return res;
	}
};

}
}

#endif
