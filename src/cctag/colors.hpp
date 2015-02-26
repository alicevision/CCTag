#ifndef _CCTAG_COLORS_HPP_
#define	_CCTAG_COLORS_HPP_

#include <cctag/geometry/point.hpp>

#include <boost/array.hpp>

namespace rom {

struct Color : public boost::array<double, 4>
{
	Color()
	{
		(*this)[0] = 0.0;
		(*this)[1] = 0.0;
		(*this)[2] = 0.0;
		(*this)[3] = 0.0;
	}
	Color( const double r, const double g, const double b, const double alpha )
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
	inline double  r() const { return (*this)[ 0 ]; }
	inline double& r()       { return (*this)[ 0 ]; }

	inline double  g() const { return (*this)[ 1 ]; }
	inline double& g()       { return (*this)[ 1 ]; }

	inline double  b() const { return (*this)[ 2 ]; }
	inline double& b()       { return (*this)[ 2 ]; }

	inline double  alpha() const { return (*this)[ 3 ]; }
	inline double& alpha()       { return (*this)[ 3 ]; }

	inline void setR( const double r ) { this->r() = r; }
	inline void setG( const double g ) { this->g() = g; }
	inline void setB( const double b ) { this->b() = b; }
	inline void setAlpha( const double a ) { this->alpha() = a; }
};

static const Color color_white( 1.0, 1.0, 1.0, 1.0 );
static const Color color_red( 1.0, 0.0, 0.0, 1.0 );
static const Color color_green( 0.0, 1.0, 0.0, 1.0 );
static const Color color_blue( 0.0, 0.0, 1.0, 1.0 );

}

#endif
