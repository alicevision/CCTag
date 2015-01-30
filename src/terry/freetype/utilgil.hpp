#ifndef _TERRY_FREETYPE_UTIL_HPP_
#define _TERRY_FREETYPE_UTIL_HPP_

// (C) Copyright Tom Brinkman 2007.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)

#include <boost/gil/image.hpp>
#include <terry/numeric/operations.hpp>
#include <terry/numeric/assign.hpp>
#include <terry/numeric/operations_assign.hpp>

namespace terry {

enum
{
	Left    = ( 0x1 << 0 ),
	Center  = ( 0x1 << 1 ),
	Right   = ( 0x1 << 2 ),
	Top     = ( 0x1 << 3 ),
	Middle  = ( 0x1 << 4 ),
	Bottom  = ( 0x1 << 5 ),
	FMiddle = ( 0x1 << 6 ),
	FBottom = ( 0x1 << 7 ),
};

struct make_special_interval
{
	int size, r, adj, pos;

	make_special_interval( int width, int size )
		: size( size )
		, r( ( width - 1 ) % size )
		, adj( ( width - 1 ) / size )
		, pos( 0 ) {}

	int operator()( int in )
	{
		int out = pos;

		pos += adj;

		if( r )
		{
			++pos;
			--r;
		}

		return out;
	}

};

struct make_alpha_blend
{
	double alpha;

	make_alpha_blend( double alpha ) : alpha( alpha ) {}

	template <typename T>
	void operator()( T& dst, const T src )
	{
		double dbl = ( dst * alpha - src * alpha + src * 255.0 ) / 255.0;

		dst = (int) dbl;
	}

};

template<typename GlyphView, typename View>
inline
void copy_and_convert_alpha_blended_pixels( const GlyphView& glyphView, const typename View::value_type& glyphColor, const View& dstView )
{
	using namespace boost::gil;
	//	typedef typename GlyphView::value_type GlyphPixel;
	typedef typename View::value_type Pixel;

	//	typedef pixel<bits32f, layout<typename color_space_type<GlyphView>::type> > GlyphPixel32f;
	//	typedef pixel<typename channel_type<view_t>::type, layout<gray_t> > PixelGray;

	//	BOOST_STATIC_ASSERT(( boost::is_same<typename color_space_type<glyphView>::type, gray_t>::value ));
	//	BOOST_STATIC_ASSERT(( boost::is_same<typename channel_type<glyphView>::type, bits32f>::value ));

	for( int y = 0;
	     y < dstView.height();
	     ++y )
	{
		typename GlyphView::x_iterator it_glyph = glyphView.x_at( 0, y );
		typename View::x_iterator it_img        = dstView.x_at( 0, y );
		for( int x = 0;
		     x < dstView.width();
		     ++x, ++it_glyph, ++it_img )
		{
			Pixel pColor = glyphColor;
			numeric::pixel_multiplies_scalar_assign_t<float, Pixel>()
			(
			    get_color( *it_glyph, gray_color_t() ),
			    pColor
			);
			numeric::pixel_multiplies_scalar_assign_t<float, Pixel>()
			(
			    channel_invert( get_color( *it_glyph, gray_color_t() ) ),
			    *it_img
			);
			numeric::pixel_plus_assign_t<Pixel, Pixel>()
			(
			    pColor,
			    *it_img
			);
//			TUTTLE_COUT_VAR( get_color( *it_glyph, gray_color_t() ) );
//			TUTTLE_COUT_VAR( (*it_img)[0] );
//			TUTTLE_COUT_VAR( (int)(color[0]) );
//			TUTTLE_COUT_VAR( (int)(pColor[0]) );
		}
	}
}

template <typename view_t, typename value_type>
inline void wuline( const view_t& view, const value_type& pixel,
                    int X0, int Y0, int X1, int Y1,
                    int NumLevels, int IntensityBits )
{
	unsigned short IntensityShift, ErrorAdj, ErrorAcc;
	unsigned short ErrorAccTemp, Weighting, WeightingComplementMask;
	short DeltaX, DeltaY, Temp, XDir;

	if( Y0 > Y1 )
	{
		Temp = Y0;
		Y0   = Y1;
		Y1   = Temp;
		Temp = X0;
		X0   = X1;
		X1   = Temp;
	}

	view( X0, Y0 ) = pixel;

	if( ( DeltaX = X1 - X0 ) >= 0 )
	{
		XDir = 1;
	}
	else
	{
		XDir   = -1;
		DeltaX = -DeltaX;
	}

	if( ( DeltaY = Y1 - Y0 ) == 0 )
	{
		while( DeltaX-- != 0 )
		{
			X0            += XDir;
			view( X0, Y0 ) = pixel;
		}

		return;
	}

	if( DeltaX == 0 )
	{
		do
		{
			Y0++;
			view( X0, Y0 ) = pixel;
		}
		while( --DeltaY != 0 );

		return;
	}

	if( DeltaX == DeltaY )
	{
		do
		{
			X0 += XDir;
			Y0++;
			view( X0, Y0 ) = pixel;
		}
		while( --DeltaY != 0 );

		return;
	}

	ErrorAcc                = 0;
	IntensityShift          = 16 - IntensityBits;
	WeightingComplementMask = NumLevels - 1;

	if( DeltaY > DeltaX )
	{
		ErrorAdj = ( (unsigned long) DeltaX << 16 ) / (unsigned long) DeltaY;

		while( --DeltaY )
		{
			ErrorAccTemp = ErrorAcc;
			ErrorAcc    += ErrorAdj;

			if( ErrorAcc <= ErrorAccTemp )
				X0 += XDir;

			Y0++;

			Weighting = ErrorAcc >> IntensityShift;

			value_type dst = pixel;
			boost::gil::static_for_each( dst, view( X0, Y0 ),
			                             make_alpha_blend( ( Weighting ^ WeightingComplementMask ) ) );
			view( X0, Y0 ) = dst;

			dst = pixel;
			boost::gil::static_for_each( dst, view( X0 + XDir, Y0 ),
			                             make_alpha_blend( Weighting ) );
			view( X0 + XDir, Y0 ) = dst;
		}

		view( X1, Y1 ) = pixel;
		return;
	}

	ErrorAdj = ( (unsigned long) DeltaY << 16 ) / (unsigned long) DeltaX;
	while( --DeltaX )
	{
		ErrorAccTemp = ErrorAcc;
		ErrorAcc    += ErrorAdj;

		if( ErrorAcc <= ErrorAccTemp )
			Y0++;

		X0 += XDir;

		Weighting = ErrorAcc >> IntensityShift;

		value_type dst = pixel;
		boost::gil::static_for_each( dst, view( X0, Y0 ),
		                             make_alpha_blend( Weighting ^ WeightingComplementMask ) );
		view( X0, Y0 ) = dst;

		dst = pixel;
		boost::gil::static_for_each( dst, view( X0, Y0 + 1 ),
		                             make_alpha_blend( Weighting ) );
		view( X0, Y0 + 1 ) = dst;
	}

	view( X1, Y1 ) = pixel;
}

template <typename view_t>
struct draw_line
{
	typedef typename view_t::value_type value_type;
	const view_t& view;
	const value_type& pixel;
	short NumLevels;
	unsigned short IntensityBits;

	draw_line( const view_t& view, const value_type& pixel,
	           int NumLevels = 256, int IntensityBits = 8 )
		: view( view )
		, pixel( pixel )
		, NumLevels( NumLevels )
		, IntensityBits( IntensityBits ) {}

	template <typename point_t>
	void operator()( point_t pt0, point_t pt1 )
	{
		wuline( view, pixel,
		        pt0.x, pt0.y, pt1.x, pt1.y,
		        NumLevels, IntensityBits );
	}

};

}

#endif

