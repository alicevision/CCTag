#ifndef _TERRY_MATH_RECT_HPP_
#define _TERRY_MATH_RECT_HPP_

#include <boost/gil/utilities.hpp>

#include <cmath>
#include <algorithm>
#include <iostream>

namespace terry {

using namespace boost::gil;

template <typename T>
class Rect {
public:
    typedef T value_type;
    static const std::size_t num_dimensions = 4;

    Rect() : x1(0), y1(0), x2(0), y2(0) {}
    Rect( const T nX1, const T nY1, const T nX2, const T nY2 )
		: x1(nX1), y1(nY1)
		, x2(nX2), y2(nY2)
	{}
    Rect(const Rect& r)
		: x1(r.x1), y1(r.y1)
		, x2(r.x2), y2(r.y2)
	{}
	template<class OtherRect>
    Rect(const OtherRect& r)
		: x1(r.x1), y1(r.y1)
		, x2(r.x2), y2(r.y2)
	{}
    ~Rect() {}

    Rect& operator=(const Rect& r) { x1=r.x1; y1=r.y1; x2=r.x2; y2=r.y2; return *this; }

    const T& operator[](const std::size_t i)          const   { return this->*mem_array[i]; }
          T& operator[](const std::size_t i)                  { return this->*mem_array[i]; }

    point2<T> cornerMin() const { return point2<T>(x1, y1); }
    point2<T> cornerMax() const { return point2<T>(x2, y2); }
    
    point2<T> size() const { return point2<T>(x2-x1, y2-y1); }

    T x1,y1,x2,y2;
	
private:
    // this static array of pointers to member variables makes operator[] safe and doesn't seem to exhibit any performance penalty
    static T Rect<T>::* const mem_array[num_dimensions];
};

template <typename T>
T Rect<T>::* const Rect<T>::mem_array[Rect<T>::num_dimensions] = { &Rect<T>::x1, &Rect<T>::y1, &Rect<T>::x2, &Rect<T>::y2 };


template <typename T>
std::ostream& operator<<( std::ostream& os, const Rect<T>& rect )
{
	os << "{ "
	   << rect.x1 << ", " <<  rect.y1 << ", "
	   << rect.x2 << ", " <<  rect.y2
	   << " }";
	return os;
}

/**
 * @brief Retrieve the bounding box of an image [0, 0, width, height].
 */
template<typename T, class View>
Rect<T> getBounds( const View& v )
{
	return Rect<T>( 0, 0, v.width(), v.height() );
}

template<class Point, class Rect>
inline bool pointInRect( const Point& p, const Rect& rec )
{
	Rect orientedRec;
	if( rec.x1 < rec.x2 )
	{
		orientedRec.x1 = rec.x1;
		orientedRec.x2 = rec.x2;
	}
	else
	{
		orientedRec.x1 = rec.x2;
		orientedRec.x2 = rec.x1;
	}
	if( rec.y1 < rec.y2 )
	{
		orientedRec.y1 = rec.y1;
		orientedRec.y2 = rec.y2;
	}
	else
	{
		orientedRec.y1 = rec.y2;
		orientedRec.y2 = rec.y1;
	}
	return p.x >= orientedRec.x1 && p.x <= orientedRec.x2 &&
	       p.y >= orientedRec.y1 && p.y <= orientedRec.y2;
}


template<class Rect>
inline Rect translateRegion( const Rect& windowRoW, const Rect& dependingTo )
{
	Rect windowOutput = windowRoW;
	windowOutput.x1 -= dependingTo.x1; // to output clip coordinates
	windowOutput.y1 -= dependingTo.y1;
	windowOutput.x2 -= dependingTo.x1;
	windowOutput.y2 -= dependingTo.y1;
	return windowOutput;
}

template<class Rect, class Point>
inline Rect translateRegion( const Rect& windowRoW, const Point& move )
{
	Rect windowOutput = windowRoW;
	windowOutput.x1 += move.x;
	windowOutput.y1 += move.y;
	windowOutput.x2 += move.x;
	windowOutput.y2 += move.y;
	return windowOutput;
}

template<class Rect>
inline Rect translateRegion( const Rect& windowRoW, const std::ptrdiff_t x, const std::ptrdiff_t y )
{
	Rect windowOutput = windowRoW;
	windowOutput.x1 += x;
	windowOutput.y1 += y;
	windowOutput.x2 += x;
	windowOutput.y2 += y;
	return windowOutput;
}

template<class R>
inline R rectanglesBoundingBox( const R& a, const R& b )
{
	R res;
	res.x1 = std::min( a.x1, b.x1 );
	res.x2 = std::max( res.x1, std::max( a.x2, b.x2 ) );
	res.y1 = std::min( a.y1, b.y1 );
	res.y2 = std::max( res.y1, std::max( a.y2, b.y2 ) );
	return res;
}

template<class R>
inline R rectanglesIntersection( const R& a, const R& b )
{
	R res;
	res.x1 = std::max( a.x1, b.x1 );
	res.x2 = std::max( res.x1, std::min( a.x2, b.x2 ) );
	res.y1 = std::max( a.y1, b.y1 );
	res.y2 = std::max( res.y1, std::min( a.y2, b.y2 ) );
	return res;
}

template<class R, class V>
inline R rectangleGrow( const R& rect, const V marge )
{
	R res = rect;
	res.x1 -= marge;
	res.y1 -= marge;
	res.x2 += marge;
	res.y2 += marge;
	return res;
}

template<class R, class V>
inline R rectangleReduce( const R& rect, const V marge )
{
	R res = rect;
	res.x1 += marge;
	res.y1 += marge;
	res.x2 -= marge;
	res.y2 -= marge;
	return res;
}


}

#endif
