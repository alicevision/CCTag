/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_NUMERICAL_CIRCLE_HPP_
#define _CCTAG_NUMERICAL_CIRCLE_HPP_

#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/utils/Defines.hpp>
#include <Eigen/Dense>
#include <cmath>

namespace cctag {
namespace numerical {
namespace geometry {

class Circle : public Ellipse
{
public:

	Circle() = default;

	explicit Circle( float r );

	Circle( const Point2d<Eigen::Vector3f>& center, float r );

	template <typename T>
	Circle( const Point2d<T> & p1, const Point2d<T> & p2, const Point2d<T> & p3 )
	{
        const float x1 = p1.x();
        const float y1 = p1.y();
        const float x2 = p2.x();
        const float y2 = p2.y();
        const float x3 = p3.x();
        const float y3 = p3.y();
        const float det = ( x1 - x2 ) * ( y1 - y3 ) - ( y1 - y2 ) * ( x1 - x3 );

        if( det == 0 )
        {
            ///@todo
        }

        Eigen::Matrix2f A;
        A( 0, 0 ) = x2 - x1;
        A( 0, 1 ) = y2 - y1;
        A( 1, 0 ) = x3 - x1;
        A( 1, 1 ) = y3 - y1;

        Eigen::Vector2f bb;
        bb( 0 ) = ( x1 + x2 ) / 2 * ( x2 - x1 ) + ( y1 + y2 ) / 2 * ( y2 - y1 );
        bb( 1 ) = ( x1 + x3 ) / 2 * ( x3 - x1 ) + ( y1 + y3 ) / 2 * ( y3 - y1 );

        //auto aux = A.colPivHouseholderQr().solve(bb);
        const auto aux = A.inverse()*bb;

        const float xc = aux( 0 );
        const float yc = aux( 1 );
        const float r = sqrt( ( x1 - xc ) * ( x1 - xc ) + ( y1 - yc ) * ( y1 - yc ) );

        Point2d<Eigen::Vector3f> c( xc, yc );
        Ellipse::setParameters( c, r, r, 0.f );
	}

private:
};

}
}
}

#endif

