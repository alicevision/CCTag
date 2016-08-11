/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_COLORS_HPP_
#define	_CCTAG_COLORS_HPP_

#include <cctag/geometry/Point.hpp>

#include <boost/array.hpp>

namespace cctag {

struct Color : public boost::array<float, 4>
{
	Color()
	{
		(*this)[0] = 0.f;
		(*this)[1] = 0.f;
		(*this)[2] = 0.f;
		(*this)[3] = 0.f;
	}
	Color( const float r, const float g, const float b, const float alpha )
	{
		(*this)[0] = r;
		(*this)[1] = g;
		(*this)[2] = b;
		(*this)[3] = alpha;
	}
	
	Color operator=( const Color & color )
	{
		(*this)[0] = color[0];
		(*this)[1] = color[1];
		(*this)[2] = color[2];
		(*this)[3] = color[3];
		return (*this);
	}
	inline float  r() const { return (*this)[ 0 ]; }
	inline float& r()       { return (*this)[ 0 ]; }

	inline float  g() const { return (*this)[ 1 ]; }
	inline float& g()       { return (*this)[ 1 ]; }

	inline float  b() const { return (*this)[ 2 ]; }
	inline float& b()       { return (*this)[ 2 ]; }

	inline float  alpha() const { return (*this)[ 3 ]; }
	inline float& alpha()       { return (*this)[ 3 ]; }

	inline void setR( const float r ) { this->r() = r; }
	inline void setG( const float g ) { this->g() = g; }
	inline void setB( const float b ) { this->b() = b; }
	inline void setAlpha( const float a ) { this->alpha() = a; }
};

static const Color color_white( 1.f, 1.f, 1.f, 1.f );
static const Color color_red( 1.f, 0.f, 0.f, 1.f );
static const Color color_green( 0.f, 1.f, 0.f, 1.f );
static const Color color_blue( 0.f, 0.f, 1.f, 1.f );

}

#endif
