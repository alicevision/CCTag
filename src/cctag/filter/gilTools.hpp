/**
 * @brief Bunch of adobe GIL tools.
 *
 *
 * @file   gilTools.hpp
 * @author edubois
 *
 * @date March 17, 2010, 11:50 AM
 */

#ifndef _GILTOOLS_HPP
#define _GILTOOLS_HPP
#include <boost/gil/gil_all.hpp>

namespace boost {
namespace gil {

template <class Pixel, bool IsPlanar = false>
struct view_from_pixel
{
	typedef typename boost::gil::view_type<Pixel, typename Pixel::layout_t, IsPlanar>::type type;
};

template< class SVIEW, class DVIEW, class SPIX, class DPIX >
void convertedCopy( SPIX* sdata, DPIX* ddata, const int w, const int h, bool upsidedown = false )
{
	typedef typename channel_type< SPIX >::type sValueType;
	typedef typename channel_type< DPIX >::type dValueType;

	SVIEW svw = interleaved_view( w, h, (SPIX*)sdata, w * sizeof( sValueType ) * num_channels<SVIEW>::type::value );
	DVIEW dvw = interleaved_view( w, h, (DPIX*)ddata, w * sizeof( dValueType ) * num_channels<DVIEW>::type::value );

	// Apply upside down transform when copying
	if( upsidedown )
	{
		copy_and_convert_pixels( flipped_up_down_view( svw ), dvw );
	}
	else
	{
		copy_and_convert_pixels( svw, dvw );
	}
}

template<class SView>
rgb32f_pixel_t getAverageColor( const SView & rgbView, const std::size_t x, const std::size_t y )
{
	double r = 0.0, g = 0.0, b = 0.0;
	std::size_t n = 1;
	typename SView::xy_locator loc = rgbView.xy_at( x, y );
	{
		rgb32f_pixel_t pix;
		color_convert( *loc, pix );
		r += get_color(pix, red_t());
		g += get_color(pix, green_t());
		b += get_color(pix, blue_t());
	}
	if ( x > 0 )
	{
		{
			rgb32f_pixel_t pix;
			color_convert( loc( -1, 0 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
		if ( y > 0 )
		{
			rgb32f_pixel_t pix;
			color_convert( loc( -1, -1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
		if ( y + 1 < rgbView.height() )
		{
			rgb32f_pixel_t pix;
			color_convert( loc( -1, 1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
	}
	if ( y > 0 )
	{
		{
			rgb32f_pixel_t pix;
			color_convert( loc( 0, -1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
		if ( x + 1 < rgbView.width() )
		{
			rgb32f_pixel_t pix;
			color_convert( loc( 1, -1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
	}
	if ( x + 1 < rgbView.width() )
	{
		rgb32f_pixel_t pix;
		color_convert( loc( 1, 0 ), pix );
		r += get_color(pix, red_t());
		g += get_color(pix, green_t());
		b += get_color(pix, blue_t());
		++n;
	}
	if ( y + 1 < rgbView.height() )
	{
		{
			rgb32f_pixel_t pix;
			color_convert( loc( 0, 1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
		if ( x + 1 < rgbView.width() )
		{
			rgb32f_pixel_t pix;
			color_convert( loc( 1, 1 ), pix );
			r += get_color(pix, red_t());
			g += get_color(pix, green_t());
			b += get_color(pix, blue_t());
			++n;
		}
	}
	rgb32f_pixel_t pix;
	get_color(pix, red_t()) = r / n;
	get_color(pix, green_t()) = g / n;
	get_color(pix, blue_t()) = b / n;
	return pix;
}

struct black_filler
{
	template< class P>
	inline P operator()( const P& p ) const
	{
		P p2;

		for( int v = 0; v < num_channels<P>::type::value; ++v )
		{
			p2[v] = 0;
		}
		return p2;
	}

};

/**
 * @brief Remplit une image en noir (tous les canneaux a 0 sauf la couche alpha a 1 (ou 255, ou ...))
 */
template <class View>
void fill_black( View& v )
{
	transform_pixels( v, v, black_filler() );
	// Following doesn't work for built-in pixel types
	//	fill_pixels( v, get_black( v ) );
}

struct white_filler
{
	template< class P>
	inline P operator()( const P& p ) const
	{
		P p2;

		for( int v = 0; v < num_channels<P>::type::value; ++v )
		{
			p2[v] = channel_traits< typename channel_type< P >::type >::max_value();
		}
		return p2;
	}

};

template <class View>
void fill_white( View& v )
{
	transform_pixels( v, v, white_filler() );
	// Following doesn't work for built-in pixel types
	//	fill_pixels( v, get_black( v ) );
}


} // namespace gil
} // namespace boost

#endif

