#ifndef _TERRY_COLOR_COLORSPACE_RGB_HPP_
#define	_TERRY_COLOR_COLORSPACE_RGB_HPP_

#include "xyz.hpp"

#include <boost/gil/rgba.hpp>

namespace terry {
namespace color {

namespace rgb_colorspace
{
	using ::boost::gil::red_t;
	using ::boost::gil::green_t;
	using ::boost::gil::blue_t;
}
typedef ::boost::gil::rgb_t rgb_colorspace_t;
using ::boost::gil::rgb_layout_t;

struct RGBParams : public IColorParams
{
	typedef RGBParams This;
	virtual bool operator==( const IColorParams& other ) const
	{
		const This* other_ptr = dynamic_cast<const This*>(&other);
		return other_ptr != NULL;
	}
};

/**
 * @brief RGB colorspace description
 * @todo
 */
struct RGB
{
	typedef XYZ reference;
	typedef RGBParams params;
	
	typedef rgb_colorspace_t colorspace;
	typedef rgb_layout_t layout;
};

template<typename SChannelType, typename DChannelType>
void color_transform( const RGBParams& params, const pixel<SChannelType,RGB::layout>& src, pixel<DChannelType,XYZ::layout>& dst )
{
	dst = terry::get_black< pixel<DChannelType,XYZ::layout> >();
}
template<typename SChannelType, typename DChannelType>
void color_transform( const RGBParams& params, const pixel<SChannelType,XYZ::layout>& src, pixel<DChannelType,RGB::layout>& dst )
{
	dst = terry::get_black< pixel<DChannelType,RGB::layout> >();
}

}
}

#endif

