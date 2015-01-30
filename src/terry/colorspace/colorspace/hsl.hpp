#ifndef _TERRY_COLOR_COLORSPACE_HSL_HPP_
#define	_TERRY_COLOR_COLORSPACE_HSL_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {

/// \addtogroup ColorNameModel
/// \{
namespace hsl_colorspace
{
/// \brief Hue
struct hue_t {};    
/// \brief Saturation
struct saturation_t {};
/// \brief Lightness
struct lightness_t {}; 
}
/// \}

/// \ingroup ColorSpaceModel
typedef ::boost::mpl::vector3<
		hsl_colorspace::hue_t,
		hsl_colorspace::saturation_t,
		hsl_colorspace::lightness_t
	> hsl_colorspace_t;

/// \ingroup LayoutModel
typedef ::boost::gil::layout<hsl_colorspace_t> hsl_layout_t;


struct HSLParams : public IColorParams
{
	typedef HSLParams This;
	virtual bool operator==( const IColorParams& other ) const
	{
		const This* other_ptr = dynamic_cast<const This*>(&other);
		return other_ptr != NULL;
	}
};
/**
 * @brief HSL colorspace description
 * @todo
 */
struct HSL
{
	typedef RGB reference;
	typedef HSLParams params;
	
	typedef hsl_colorspace_t colorspace;
	typedef hsl_layout_t layout;
};

template<typename SChannelType, typename DChannelType>
void color_transform( const HSLParams& params, const pixel<SChannelType,HSL::layout>& src, pixel<DChannelType,RGB::layout>& dst )
{
	pixel_zeros( dst );
}
template<typename SChannelType, typename DChannelType>
void color_transform( const HSLParams& params, const pixel<SChannelType,RGB::layout>& src, pixel<DChannelType,HSL::layout>& dst )
{
	typedef typename floating_channel_type_t<DChannelType>::type ChannelFloat;
	
	// only ChannelFloat for hsl is supported
	ChannelFloat temp_red   = channel_convert<ChannelFloat>( get_color( src, red_t()   ));
	ChannelFloat temp_green = channel_convert<ChannelFloat>( get_color( src, green_t() ));
	ChannelFloat temp_blue  = channel_convert<ChannelFloat>( get_color( src, blue_t()  ));

	ChannelFloat hue, saturation, lightness;

	ChannelFloat min_color = std::min( temp_red, std::min( temp_green, temp_blue ));
	ChannelFloat max_color = std::max( temp_red, std::max( temp_green, temp_blue ));

	ChannelFloat diff = max_color - min_color;

	if( diff == 0.0 )
	{
		// rgb color is gray
		get_color( dst, hsl_colorspace::hue_t() ) = 0;
		get_color( dst, hsl_colorspace::saturation_t() ) = 0;
		// doesn't matter which rgb channel we use, they all have the same value.
		get_color( dst, hsl_colorspace::lightness_t() )	= channel_convert<DChannelType>( temp_red );
		return;
	}
	
	// lightness calculation
	lightness = ( min_color + max_color ) * 0.5f;

	// saturation calculation
	if( lightness < 0.5f )
	{
		saturation = diff / ( min_color + max_color );
	}
	else
	{
		saturation = ( max_color - min_color ) / ( 2.f - diff );
	}

	// hue calculation
	if( max_color == temp_red )
	{
		// max_color is red
		hue = (double)( temp_green - temp_blue ) / diff;

	}
	else if( max_color == temp_green )
	{
		// max_color is green
		// 2.0 + (b - r) / (maxColor - minColor);
		hue = 2.f + (double)( temp_blue - temp_red ) / diff;
	}
	else
	{
		// max_color is blue
		hue = 4.f + (double)( temp_red - temp_green ) / diff;
	}

	if( hue < 0.f )
	{
		hue += 6.f;
	}
	hue /= 6.f;

	get_color( dst, hsl_colorspace::hue_t()        ) = channel_convert<DChannelType>( hue );
	get_color( dst, hsl_colorspace::saturation_t() ) = channel_convert<DChannelType>( saturation );
	get_color( dst, hsl_colorspace::lightness_t()  ) = channel_convert<DChannelType>( lightness );
}

BOOST_MPL_ASSERT( ( ::boost::mpl::equal<
					::boost::mpl::vector<XYZ, RGB, HSL>,
					color_dependencies<HSL>::from_root
					> ) );

BOOST_MPL_ASSERT( ( ::boost::mpl::equal<
					::boost::mpl::vector<HSL, RGB, XYZ>,
					color_dependencies<HSL>::to_root
					> ) );

BOOST_MPL_ASSERT(( ::boost::mpl::equal<
                                        ::boost::mpl::vector<XYZ, RGB>,
                                        ::boost::mpl::at_c<color_dependencies<HSL>::color_steps_from_root, 0>::type
                                        > ));
BOOST_MPL_ASSERT(( ::boost::mpl::equal<
                                        ::boost::mpl::vector<RGB, HSL>,
                                        ::boost::mpl::at_c<color_dependencies<HSL>::color_steps_from_root, 1>::type
                                        > ));


}

TERRY_DEFINE_GIL_INTERNALS_3(hsl)
TERRY_DEFINE_COLORSPACE_STANDARD_TYPEDEFS(hsl)

}

#endif

