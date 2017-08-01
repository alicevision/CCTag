/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_NUMERICAL_ELLIPSEFROMPOINTS_HPP_
#define _CCTAG_NUMERICAL_ELLIPSEFROMPOINTS_HPP_

#include <cctag/geometry/Ellipse.hpp>

#include <cstddef>
#include <cmath>
#include <vector>


namespace cctag {
namespace numerical {
namespace geometry {

Point2d<Eigen::Vector3f> extractEllipsePointAtAngle( const Ellipse & ellipse, float theta );

Point2d<Eigen::Vector3f> pointOnEllipse( const Ellipse & ellipse, const Point2d<Eigen::Vector3f> & p );

///@todo rename this function
void points( const Ellipse & ellipse, std::size_t nb, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );
///@todo rename this function
void points( const Ellipse & ellipse, std::size_t nb, float phi1, float phi2, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );


// Direct implementation of "NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES" by Halir, Flusser
// Fitzgibbon's paper "Direct Least Squares Fitting of Ellipses" was previously THE method used in opencv; see commit
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
template<typename It> // It must be an iterator pointing to something derived from Eigen::Matrix<T, 3, 1>, _or a pointer to it_
void fitEllipse(It begin, It end, Ellipse& e);

template <class T>
void fitEllipse(const std::vector<cctag::Point2d<T>> & points, Ellipse& e)
{
  fitEllipse(points.begin(), points.end(), e);
}

void ellipsePoint( const cctag::numerical::geometry::Ellipse& ellipse, float theta, Eigen::Vector3f& pt );
void computeIntermediatePoints(const Ellipse & ellipse, Point2d<Eigen::Vector3i> & pt11, Point2d<Eigen::Vector3i> & pt12, Point2d<Eigen::Vector3i> & pt21, Point2d<Eigen::Vector3i> & pt22);
void rasterizeEllipticalArc(const Ellipse & ellipse, const Point2d<Eigen::Vector3i> & pt1, const Point2d<Eigen::Vector3i> & pt2, std::vector< Point2d<Eigen::Vector3i> > & vPoint, std::size_t intersectionIndex);
void rasterizeEllipse( const Ellipse & ellipse, std::vector< Point2d<Eigen::Vector3i> > & vPoint );

/**
 * Get the pixel perimeter of an ellipse
 * @param ellipse
 * @return the perimeter in number of pixels
 */
std::size_t rasterizeEllipsePerimeter( const Ellipse & ellipse );

/**
 * @brief Compute intersections if any, between a line of equation Y = y and an ellipse.
 *
 * @param[in] ellipse
 * @param[in] y ordonate for which we compute intersection abscissa
 * @return intersected points sorted in ascend order (returns x coordinates: 0, 1, or 2 points).
 */
std::vector<float> intersectEllipseWithLine( const numerical::geometry::Ellipse& ellipse, float y, bool horizontal);

}
}
}

#endif

