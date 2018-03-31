/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/Circle.hpp>
#include <cctag/geometry/Point.hpp>


namespace cctag {
namespace numerical {
namespace geometry {

Circle::Circle( const Point2d<Eigen::Vector3f>& center, float r )
	: Ellipse( center, r, r, 0.f )
{
}

Circle::Circle( float r )
	: Ellipse( Point2d<Eigen::Vector3f>(0.f, 0.f) , r, r, 0.f )
{
}

}
}
}
