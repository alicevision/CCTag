#ifndef _TERRY_FREETYPE_FREEGIL_HPP_
#define _TERRY_FREETYPE_FREEGIL_HPP_

// (C) Copyright Tom Brinkman 2007.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt.)

#include "utilgil.hpp"
#include "utilstl.hpp"

#include <terry/math/Rect.hpp>
#include <terry/geometry/subimage.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/cast.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>
#include <boost/spirit/include/classic_lists.hpp>

#include <string>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H


namespace terry {

using namespace boost::gil;

struct make_metric
{

	template <typename glyph_t >
	FT_Glyph_Metrics operator()( const glyph_t& glyph )
	{
		BOOST_ASSERT( glyph.face );
		int load_flags = FT_LOAD_DEFAULT;
		int index      = FT_Get_Char_Index( glyph.face, glyph.ch );
		FT_Load_Glyph( glyph.face, index, load_flags );
		return glyph.face->glyph->metrics;
	}

};

struct make_kerning
{
	int left_glyph;

	make_kerning() : left_glyph( 0 ) {}

	template <typename glyph_t>
	int operator()( const glyph_t& glyph )
	{
		int right_glyph = FT_Get_Char_Index( glyph.face, glyph.ch );

		if( !FT_HAS_KERNING( glyph.face ) || !left_glyph || !right_glyph )
			return 0;

		FT_Vector delta;
		FT_Get_Kerning( glyph.face, left_glyph, right_glyph, FT_KERNING_DEFAULT, &delta );
		left_glyph = right_glyph;
		return delta.x >> 6;
	}

};

struct make_width
{
	int advance;
	int lastwidth;
	int lastadvance;

	make_width() : advance( 0 )
		, lastwidth( 0 )
		, lastadvance( 0 ) {}

	operator int()
	{
		return advance - ( lastadvance - lastwidth );
	}

	void operator()( FT_Glyph_Metrics metrics, int kerning )
	{
		lastadvance = kerning + ( metrics.horiAdvance >> 6 );
		lastwidth   = ( metrics.width >> 6 );
		advance    += lastadvance;
	}

};

struct make_advance_width
{
	int advance;

	make_advance_width() : advance( 0 ) {}

	operator int()
	{
		return advance;
	}

	void operator()( FT_Glyph_Metrics metrics, int kerning )
	{
		advance += kerning + ( metrics.horiAdvance >> 6 );
	}

};

struct make_advance_height
{
	int height;

	make_advance_height() : height( 0 ) {}

	operator int()
	{
		return height;
	}

	void operator()( FT_Glyph_Metrics metrics )
	{
		int advance = ( metrics.vertAdvance >> 6 );

		height = ( std::max )( height, advance );
	}

};

struct make_height
{
	int height;

	make_height() : height( 0 ) {}

	operator int()
	{
		return height;
	}

	void operator()( FT_Glyph_Metrics metrics )
	{
		int h = ( metrics.height >> 6 );

		height = ( std::max )( height, h );
	}

};

struct make_glyph_height
{
	int height;

	make_glyph_height() : height( 0 ) {}

	operator int()
	{
		return height;
	}

	void operator()( FT_Glyph_Metrics metrics )
	{
		int n = ( metrics.height >> 6 ) - ( metrics.horiBearingY >> 6 );

		height = ( std::max )( height, n );
	}

};

template <typename view_t>
class render_glyph
{
public:
	typedef render_glyph<view_t> This;
	typedef typename view_t::value_type Pixel;
	typedef Rect<std::ptrdiff_t> rect_t;
	typedef point2<std::ptrdiff_t> point_t;

private:
	const view_t& _outView;
	const Pixel _color;
	const double _letterSpacing;
	const rect_t _roi;
	int _x;


	//	render_glyph( const This& );

public:
	render_glyph( const view_t& outView, const Pixel& color, const double letterSpacing )
		: _outView( outView )
		, _color( color )
		, _letterSpacing( letterSpacing )
		, _roi( 0, 0, outView.width(), outView.height() )
		, _x( 0 )
		{}

	render_glyph( const view_t& outView, const Pixel& color, const double letterSpacing, const Rect<std::ptrdiff_t> roi )
		: _outView( outView )
		, _color( color )
		, _letterSpacing( letterSpacing )
		, _roi( roi )
		, _x( 0 )
		{}

	template <typename glyph_t>
	void operator()( const glyph_t& glyph, int kerning = 0 )
	{
		_x += kerning;

		FT_GlyphSlot slot = glyph.face->glyph;

		const int load_flags = FT_LOAD_DEFAULT;
		const int index      = FT_Get_Char_Index( glyph.face, glyph.ch );
		FT_Load_Glyph( glyph.face, index, load_flags );
		FT_Render_Glyph( slot, FT_RENDER_MODE_NORMAL );

		const int y        = _outView.height() - ( glyph.face->glyph->metrics.horiBearingY >> 6 );
		const int width    = glyph.face->glyph->metrics.width >> 6;
		const int height   = glyph.face->glyph->metrics.height >> 6;
		const int xadvance = glyph.face->glyph->advance.x >> 6;

		BOOST_ASSERT( width == slot->bitmap.width );
		BOOST_ASSERT( height == slot->bitmap.rows );

		const rect_t glyphRod( _x, y, _x + width, y + height );
		
		BOOST_ASSERT( glyphRod.x1 >= 0 );
		BOOST_ASSERT( glyphRod.y1 >= 0 );
		BOOST_ASSERT( glyphRod.x2 >= 0 );
		BOOST_ASSERT( glyphRod.y2 >= 0 );
		
		const rect_t glyphRoi = rectanglesIntersection( glyphRod, _roi );
		const point_t glyphRegionSize = glyphRoi.size();
		
		//TUTTLE_TCOUT_VAR( glyphRod );
		//TUTTLE_TCOUT_VAR( _roi );
		//TUTTLE_TCOUT_VAR( glyphRoi );
		//TUTTLE_TCOUT_VAR2( _x, y );
		//TUTTLE_TCOUT_VAR2( width, height );
		
		if( glyphRegionSize.x != 0 &&
		    glyphRegionSize.y != 0 )
		{
			const rect_t glyphLocalRoi = translateRegion( glyphRoi, - glyphRod.x1, - glyphRod.y1 );
			//TUTTLE_TCOUT_VAR( glyphLocalRoi );
			
			BOOST_ASSERT( glyphLocalRoi.x1 >= 0 );
			BOOST_ASSERT( glyphLocalRoi.y1 >= 0 );
			BOOST_ASSERT( glyphLocalRoi.x2 <= width );
			BOOST_ASSERT( glyphLocalRoi.y2 <= height );
			
			gray8c_view_t glyphView = interleaved_view( width, height, reinterpret_cast<gray8_pixel_t*>( slot->bitmap.buffer ), sizeof(unsigned char) * slot->bitmap.width );
			
			gray8c_view_t glyphViewRoi = subimage_view( glyphView, glyphLocalRoi );
			
			//view_t outView = subimage_view( _outView, _x, y, width, height );
			view_t outViewRoi = subimage_view( _outView, glyphRoi );
			
			BOOST_ASSERT( glyphViewRoi.width() == outViewRoi.width() );
			BOOST_ASSERT( glyphViewRoi.height() == outViewRoi.height() );

			copy_and_convert_alpha_blended_pixels( color_converted_view<gray32f_pixel_t>( glyphViewRoi ), _color, outViewRoi );
		}
		
		_x += xadvance;
		_x += _letterSpacing;
	}

};

struct find_last_fitted_glyph
{
	int width, x;

	find_last_fitted_glyph( int width ) : width( width )
		, x( 0 ) {}

	bool operator()( FT_Glyph_Metrics metric, int kerning )
	{
		x += kerning;
		int tmp = x + ( metric.width >> 6 );
		x += ( metric.horiAdvance >> 6 );
		return tmp > width;
	}

};

}


#endif

