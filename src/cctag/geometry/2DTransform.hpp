/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_2DTRANSFORM_HPP_
#define _CCTAG_2DTRANSFORM_HPP_

#include <cctag/geometry/Ellipse.hpp>
#include <Eigen/Core>
#include <vector>

namespace cctag {
namespace viewGeometry {

void projectiveTransform( const Eigen::Matrix3f& tr, cctag::numerical::geometry::Ellipse& ellipse );

} // namespace viewGeometry
} // namespace cctag

#endif

