/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/geometry/2DTransform.hpp>

namespace cctag {

namespace viewGeometry {

void projectiveTransform( const Eigen::Matrix3f& tr, cctag::numerical::geometry::Ellipse& ellipse )
{
  ellipse.setMatrix(tr.transpose() * ellipse.matrix() * tr);
}

}
}
