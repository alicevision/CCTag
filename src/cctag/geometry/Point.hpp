/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_POINT_HPP_
#define	_CCTAG_POINT_HPP_

#include <cctag/utils/Exceptions.hpp>

#include <Eigen/Core>

namespace cctag {


/**
 * @brief A 2D normalized point [x, y, 1].
 * Container: Eigen::Vector3f
 */
template<typename Container>
class Point2d : public Container
{
public:
        using Scalar = typename Container::Scalar;

        Point2d() = default;
        Point2d(const Point2d&) = default;
        Point2d(const Container& c) : Container(c)
        {
          toNonHomogen();
        }
        
	Point2d(Scalar px, Scalar py) : Container(px, py, 1)
	{
        }

	Scalar  x() const { return (*this)( 0 ); }
	Scalar& x()       { return (*this)( 0 ); }
	Scalar  y() const { return (*this)( 1 ); }
	Scalar& y()       { return (*this)( 1 ); }
	Scalar  w() const { return (*this)( 2 ); }
	Scalar& w()       { return (*this)( 2 ); }

	Point2d& toNonHomogen()
	{
		// @TODO make it more robust using fabs < epsilon
		if( w() == 0 )
		{
			throw std::invalid_argument("Normalization of an infinite point !");
		}
		x() /= w();
		y() /= w();
		w() = 1.f;
		return *this;
	}
};

/**
 * @brief A 2D normalized point [x, y, 1] + its gradient bounded_vector<T,2>
 */
template<class T>
class DirectedPoint2d : public Point2d<T>
{
        using Scalar = typename Point2d<T>::Scalar;
	using Parent = Point2d<T>;
	using This = DirectedPoint2d<T>;
        
        Eigen::Vector2f _grad;

public:
	DirectedPoint2d()
		: Point2d<T>(), _grad(0,0)
	{
        }
                
        DirectedPoint2d(const This& p) = default;

	DirectedPoint2d( const Parent& p, float dX, float dY)
		: Point2d<T>( p.x(), p.y() )
	{
          _grad(0) = dX;
          _grad(1) = dY;
        }

	DirectedPoint2d( const Scalar & px, const Scalar & py, float dX, float dY)
		: Point2d<T>( px, py )
	{
          _grad(0) = dX;
          _grad(1) = dY;
        }
        
        float  dX() const { return _grad(0); }
        float  dY() const { return _grad(1); }
        float& dX() { return _grad(0); }
        float& dY() { return _grad(1); }
        
        const Eigen::Vector2f& gradient() const 
        {
          return _grad;
        }
};

/*
template<class T>
inline size_t hash_value( const Point2dH<T> & v )
{
	size_t h = 0xdeadbeef;
	boost::hash_combine( h, v.x() );
	boost::hash_combine( h, v.y() );
	boost::hash_combine( h, v.w() );
	return h;
}
*/

}

#endif

