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


/** @brief Point voting
 * @param svw source image view
 * @param winners resulting voting winners
 */
//void vote( std::vector<EdgePoint> & points, std::vector<EdgePoint*> & candidates, const EdgePointsImage & edgesMap, WinnerMap& winners, const std::size_t searchDistance, const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns, const std::size_t minVotesToSelectCandidate );


/** @brief Point voting considering direction of the gradient for each point, v.2 of the previous vote function.
 * @param
 */

 void vote( std::vector<EdgePoint> & points, std::vector<EdgePoint*> & candidates, const EdgePointsImage & edgesMap, WinnerMap& winners, const std::size_t searchDistance, const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns, const std::size_t minVotesToSelectCandidate, const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX, const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY );
 
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

