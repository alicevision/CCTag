#ifndef _TERRY_COLOR_LAYOUT_HPP_
#define	_TERRY_COLOR_LAYOUT_HPP_

#include "colorspace.hpp"

namespace terry {
namespace color {

/**
 * @bief All supported layouts
 */
namespace layout {

/// \ingroup LayoutModel
typedef boost::gil::layout<colorspace::rgb_t> rgb;
typedef boost::gil::layout<colorspace::rgba_t> rgba;
typedef boost::gil::layout<colorspace::yuv_t> yuv;
typedef boost::gil::layout<colorspace::YPbPr_t> YPbPr;
typedef boost::gil::layout<colorspace::hsv_t> hsv;
typedef boost::gil::layout<colorspace::hsl_t> hsl;
typedef boost::gil::layout<colorspace::lab_t> lab;
typedef boost::gil::layout<colorspace::luv_t> luv;
typedef boost::gil::layout<colorspace::XYZ_t> XYZ;
typedef boost::gil::layout<colorspace::Yxy_t> Yxy;

}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template< typename Pixel,
          class IN,
          class OUT >
struct pixel_color_layout_t
{
	typedef typename channel_type<Pixel>::type Channel;
	const IN&  _in;
	const OUT& _out;

	pixel_color_layout_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	Pixel& operator()( const Pixel& p1,
	                   Pixel& p2 ) const
	{
		static_for_each(
				p1, p2,
				channel_color_layout_t< Channel, IN, OUT >( _in, _out )
			);
		return p2;
	}
};

template< class IN,
          class OUT >
struct transform_pixel_color_layout_t
{
	const IN&  _in;
	const OUT& _out;

	transform_pixel_color_layout_t( const IN& in, const OUT& out )
	: _in(in)
	, _out(out)
	{}

	template< typename Pixel>
	Pixel operator()( const Pixel& p1 ) const
	{
		Pixel p2;
		pixel_color_layout_t<Pixel, IN, OUT>( _in, _out )( p1, p2 );
		return p2;
	}
};

/**
 * @example layout_convert_view( srcView, dstView, layout::rgb(), layout::hsl() );
 */
template<class LayoutIN, class LayoutOUT, class View>
void layout_convert_view( const View& src, View& dst, const LayoutIN& layoutIn = LayoutIN(), const LayoutOUT& layoutOut = LayoutOUT() )
{
	boost::gil::transform_pixels( src, dst, transform_pixel_color_layout_t<LayoutIN, LayoutOUT>( layoutIn, layoutOut ) );
}

/**
 * @example layout_convert_pixel( srcPix, dstPix, layout::rgb(), layout::hsl() );
 */
template<class LayoutIN, class LayoutOUT, class Pixel>
void layout_convert_pixel( const Pixel& src, Pixel& dst, const LayoutIN& layoutIn = LayoutIN(), const LayoutOUT& layoutOut = LayoutOUT() )
{
	pixel_color_layout_t<Pixel, LayoutIN, LayoutOUT>( layoutIn, layoutOut )( src, dst );
}



}
}

#endif

