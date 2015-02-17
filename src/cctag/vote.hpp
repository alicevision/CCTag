#ifndef _ROM_VISION_CCTAG_VOTE_HPP_
#define _ROM_VISION_CCTAG_VOTE_HPP_

#include "EdgePoint.hpp"
#include "types.hpp"
#include "Candidate.hpp"
#include <cctag/geometry/Ellipse.hpp>

#include <boost/unordered/unordered_set.hpp>

#include <cstddef>
#include <list>
#include <utility>
#include <vector>


namespace rom {
namespace vision {
namespace marker {
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
 * winners: map associating all seeds to their voters
 * maxSearchDistance: maximum distance (in pixels) of research from one edge points
 * to another one. maximum length of a arc segment composing the polygonal line.
 * maxVotingAngle: maximum angle between of gradient directions of two consecutive
 * edge points.
 * maxVotingAngle: maximum distance ratio between of gradient directions of two consecutive
 * edge points.
 * minVotesToSelectCandidate: minimum number of received votes to select an edge 
 * point as a new seed.
 * cannyGradX: X derivative of the gray image
 * cannyGradY: Y derivative of the gray image
 */
 void vote( std::vector<EdgePoint> & points, std::vector<EdgePoint*> & seeds, const EdgePointsImage & edgesMap, WinnerMap& winners, const std::size_t searchDistance, const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns, const std::size_t minVotesToSelectCandidate, const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX, const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY );
 
/** @brief Retrieve all connected edges.
 * @param[out] convexEdgeSegment
 */
void edgeLinking( const EdgePointsImage& img, std::list<EdgePoint*>& convexEdgeSegment, EdgePoint* pmax, 
	WinnerMap& winners, std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin);

/** @brief Edge linking in a given direction
 * @param edges resulting edges sorted points
 */
void edgeLinkingDir( const EdgePointsImage& img, boost::unordered_set<std::pair<int, int> >& processed,
	EdgePoint* p, const int dir, std::list<EdgePoint*>& convexEdgeSegment,
	WinnerMap& winners, std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin);

/** @brief Concaten all childrens of each points
 * @param edges list of edges
 * @param childrens resulting childrens
 */
void childrensOf( std::list<EdgePoint*>& edges, WinnerMap& winnerMap, std::list<EdgePoint*>& childrens );

/** @brief Concaten all childrens of each points
 * @param [in/out] edges list of childrens points (from a winner)
 */
void outlierRemoval( const std::list<EdgePoint*>& childrens, std::vector<EdgePoint*>& filteredChildrens, double & SmFinal, double threshold, std::size_t weightedType = 0 );
//void outlierRemoval( std::vector<EdgePoint*>& childrens, double & SmFinal, double threshold, std::size_t weightedType = 0 ); //todo@Lilian : templater le outlierRemoval

/** @brief Search for another segment after the ellipse growinf procedure
 * @param points from the first elliptical segment
 * @param points from the candidate segment
 */
bool isAnotherSegment( numerical::geometry::Ellipse & outerEllipse, std::vector<EdgePoint*>&  outerEllipsePoints,
        const std::vector<EdgePoint*>& filteredChildrens, const Candidate & anotherCandidate,
        std::vector< std::vector< Point2dN<double> > >& cctagPoints, std::size_t numCircles,
        double thrMedianDistanceEllipse);

}
}
}
}

#endif

