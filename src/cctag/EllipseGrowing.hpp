/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_CCTAG_ELLIPSE_HPP_
#define VISION_CCTAG_ELLIPSE_HPP_

#include <cctag/Types.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>

#include <cstddef>
#include <vector>

namespace cctag
{

class CCTag;

inline bool isInEllipse(
        const cctag::numerical::geometry::Ellipse& ellipse,
        const Point2d<Eigen::Vector3f> & p)
{
  // x'Q x > 0
  auto s1 = p.dot(ellipse.matrix() * p);
  auto s2 = p.dot(ellipse.matrix() * ellipse.center());
  return s1 * s2 > 0;
  
  //return ( ( inner_prod( p, prec_prod( ellipse.matrix(), p ) ) ) *
  //         inner_prod( ellipse.center(), prec_prod( ellipse.matrix(), ellipse.center() ))
  //         > 0 );
}

/**
 * Check if ellipses overlap.
 * @param ellipse1
 * @param ellipse2
 * @return
 */
inline bool isOverlappingEllipses(
        const cctag::numerical::geometry::Ellipse& ellipse1,
        const cctag::numerical::geometry::Ellipse& ellipse2 )
{
  return ( isInEllipse( ellipse1, ellipse2.center() ) || isInEllipse( ellipse2, ellipse1.center() ) );
}

bool initMarkerCenter(
        cctag::Point2d<Eigen::Vector3f> & markerCenter,
        const std::vector< std::vector< Point2d<Eigen::Vector3f> > > & markerPoints,
        int realPixelPerimeter);

bool addCandidateFlowtoCCTag(
        EdgePointCollection& edgeCollection,
        const std::vector< EdgePoint* > & filteredChildren,
        const std::vector< EdgePoint* > & outerEllipsePoints,
        const cctag::numerical::geometry::Ellipse& outerEllipse,
        std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > >& cctagPoints,
        std::size_t numCircles);

bool ellipseGrowingInit(
        const std::vector<EdgePoint*>& filteredChildren,
        cctag::numerical::geometry::Ellipse& ellipse);

/** @brief Is a point in an elliptical hull ?
 * @param ellipse ellipse which defines the hull
 * @param delta larger of the hull
 * @param abscissa of the tested point
 * @param ordinate of the tested point
 * @return true if located between qIn and qOut false otherwise
 */
inline bool isInHull( const cctag::numerical::geometry::Ellipse& qIn, const cctag::numerical::geometry::Ellipse& qOut, const EdgePoint* p )
{
  Eigen::Vector3f pf = p->cast<float>();
  float s1 = pf.dot(qIn.matrix() * pf);
  float s2 = pf.dot(qOut.matrix() * pf);
  return s1 * s2 < 0;
  //return ( ublas::inner_prod( *p, ublas::prec_prod( qIn.matrix(), *p ) ) * ublas::inner_prod( *p, ublas::prec_prod( qOut.matrix(), *p ) ) < 0 ) ;
}

inline bool isInHull( const cctag::numerical::geometry::Ellipse& qIn, const cctag::numerical::geometry::Ellipse& qOut, const Point2d<Eigen::Vector3f>& p )
{
  float s1 = p.dot(qIn.matrix() * p);
  float s2 = p.dot(qOut.matrix() * p);
  return s1 * s2 < 0;
  //return ( ublas::inner_prod( p, ublas::prec_prod( qIn.matrix(), p ) ) * ublas::inner_prod( p, ublas::prec_prod( qOut.matrix(), p ) ) < 0 ) ;
}

inline bool isOnTheSameSide(const Point2d<Eigen::Vector3f> & p1, const Point2d<Eigen::Vector3f> &  p2, const Eigen::Vector3f& line)
{
  auto s1 = p1.dot(line);
  auto s2 = p2.dot(line);
  return s1 * s2 > 0;
  //return ( ublas::inner_prod( p1, line ) * ublas::inner_prod( p2, line ) > 0 ) ;
}

/** @brief Search recursively connected points from a point and add it in pts if it is in the ellipse hull
 * @param list of points to complete
 * @param img map of edge points
 * @param map of already processed edge points
 * @param abscissa of the point
 * @param ordinate of the point
 */
void connectedPoint( std::vector<EdgePoint*>& pts, int runId, const EdgePointCollection& img, cctag::numerical::geometry::Ellipse& qIn, cctag::numerical::geometry::Ellipse& qOut, int x, int y );

/** @brief Compute the hull from ellipse
 * @param ellipse ellipse from which the hull is computed
 * @param delta larger of the hull
 */
void computeHull( const cctag::numerical::geometry::Ellipse& ellipse, float delta, cctag::numerical::geometry::Ellipse& qIn, cctag::numerical::geometry::Ellipse& qOut );

/** @brief Ellipse hull
 * @param[in,out] pts initial points to compute all the points which are in the hull formed by the ellipse
 * which fits pt. New points will be added in pts
 * @param ellipse ellipse is an optionnal parameter if the user decide to choose his hull from an ellipse
 */
void ellipseHull( const EdgePointCollection& img, std::vector<EdgePoint*>& pts, cctag::numerical::geometry::Ellipse& ellipse, float delta, std::size_t runId);

/** @brief Ellipse growing
 * @param children vote winner children points
 * @param outerEllipsePoints outer ellipse points
 * @param ellipse target ellipse
 * @param Width of elliptic hull in ellipse growing
 */

void ellipseGrowing2( const EdgePointCollection& img, const std::vector<EdgePoint*>& filteredChildren,
                      std::vector<EdgePoint*>& outerEllipsePoints, numerical::geometry::Ellipse& ellipse,
                      float ellipseGrowingEllipticHullWidth, std::size_t runId, bool goodInit);

} // namespace cctag

#endif

