/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_CCTAG_VOTE_HPP_
#define VISION_CCTAG_VOTE_HPP_

#include <cctag/Params.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Types.hpp>
#include <cctag/Candidate.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/container/flat_set.hpp>

#include <cstddef>
#include <list>
#include <utility>
#include <vector>


namespace cctag {

/* Brief: Voting procedure. For every edge points, construct the 1st order approximation 
 * of the field line passing through it which consists in a polygonal line whose
 * extremities are two edge points.
 * Input:
 * points: set of edge points to be processed, i.e. considered as the 1st extremity
 * of the constructed field line passing through it.
 * seeds: edge points having received enough votes to be considered as a seed, i.e.
 * as an edge point belonging on an inner elliptical arc of a cctag.
 * edgesMap: map of all the edge points
 * cannyGradX: X derivative of the gray image
 * cannyGradY: Y derivative of the gray image
 */
void vote(EdgePointCollection& edgeCollection, std::vector<EdgePoint*> & seeds,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const Parameters & params);
 
/** @brief Retrieve all connected edges.
 * @param[out] convexEdgeSegment
 */
void edgeLinking(EdgePointCollection& edgeCollection, std::list<EdgePoint*>& convexEdgeSegment, EdgePoint* pmax, 
	std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin);

/** @brief Edge linking in a given direction
 * @param edges resulting edges sorted points
 */
void edgeLinkingDir(EdgePointCollection& edgeCollection, boost::container::flat_set<unsigned int>& processed,
	const EdgePoint* p, int dir, std::list<EdgePoint*>& convexEdgeSegment,
	std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin);

/** @brief Concaten all children of each points
 * @param edges list of edges
 * @param children resulting children
 */
void childrenOf(const EdgePointCollection& edgeCollection, const std::list<EdgePoint*>& edges, std::list<EdgePoint*>& children );

/** @brief Concaten all children of each points
 * @param [in/out] edges list of children points (from a winner)
 */
void outlierRemoval(
        const std::list<EdgePoint*>& children,
        std::vector<EdgePoint*>& filteredChildren,
        float & SmFinal,
        float threshold,
        std::size_t weightedType = NO_WEIGHT,
        std::size_t maxSize = std::numeric_limits<std::size_t>::max());

//void outlierRemoval( std::vector<EdgePoint*>& children, float & SmFinal, float threshold, std::size_t weightedType = 0 ); //todo@Lilian : templater le outlierRemoval

/** @brief Search for another segment after the ellipse growinf procedure
 * @param points from the first elliptical segment
 * @param points from the candidate segment
 */
bool isAnotherSegment(
        EdgePointCollection& edgeCollection,
        numerical::geometry::Ellipse & outerEllipse,
        std::vector<EdgePoint*>&  outerEllipsePoints,
        const std::vector<EdgePoint*>& filteredChildren,
        const Candidate & anotherCandidate,
        std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > >& cctagPoints,
        std::size_t numCircles,
        float thrMedianDistanceEllipse);

} // namespace cctag

#endif

