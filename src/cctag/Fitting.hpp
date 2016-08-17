/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_FITTING_H
#define _CCTAG_FITTING_H

#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Point.hpp>
#include <boost/foreach.hpp>

#include <list>
#include <string>
#include <vector>

namespace cctag {
namespace numerical {

float innerProdMin( const std::vector<cctag::EdgePoint*>& childrens, float thrCosDiffMax, Point2d<Vector3s> & p1, Point2d<Vector3s> & p2 );

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points);

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector< Point2d<Eigen::Vector3f> >& childrens );

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& childrens );

} // namespace numerical
} // namespace cctag

#endif
