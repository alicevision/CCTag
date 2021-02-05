/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Bresenham.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/utils/FileDebug.hpp>
#include <cctag/geometry/Point.hpp>
// #include <cctag/algebra/Invert.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/utils/Defines.hpp>
#include <cctag/utils/VisualDebug.hpp>

#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/multi_array/multi_array_ref.hpp>
#include <boost/multi_array/subarray.hpp>
#include <boost/container/flat_set.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <deque>
#include <array>
#include <algorithm>
#include <cmath>
#include <ostream>

#define EDGE_NOT_FOUND -1
#define CONVEXITY_LOST -2
#define LOW_FLOW -3

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
void vote(EdgePointCollection& edgeCollection,
        std::vector<EdgePoint*> & seeds,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const Parameters & params)
{
#ifdef CCTAG_VOTE_DEBUG
  std::stringstream outFilenameVote;
  outFilenameVote << "vote" << CCTagVisualDebug::instance().getPyramidLevel() << ".txt";
  CCTagFileDebug::instance().newSession(outFilenameVote.str());
#endif
  
  const int pointCount = edgeCollection.get_point_count();
  std::vector<std::vector<int>> voters;
  voters.resize(pointCount);

    for (int iEdgePoint = 0; iEdgePoint < pointCount; ++iEdgePoint ) {
        EdgePoint& p = *edgeCollection(iEdgePoint);
        EdgePoint* link;
        int ilink;
        
        link = gradientDirectionDescent(edgeCollection, p, -1, params._distSearch, dx, dy, params._thrGradientMagInVote);
        ilink = edgeCollection(link);
        edgeCollection.set_before(&p, ilink);
        
        CCTagFileDebug::instance().endVote();
        
        link = gradientDirectionDescent(edgeCollection, p, 1, params._distSearch, dx, dy, params._thrGradientMagInVote);
        ilink = edgeCollection(link);
        edgeCollection.set_after(&p, ilink);
        
        CCTagFileDebug::instance().endVote();
    }
    // Vote
    seeds.reserve(pointCount / 2);

    // todo@Lilian: remove thrVotingAngle from the parameter file
    if (params._angleVoting != 0) 
	{
		throw std::domain_error("thrVotingAngle must be equal to 0 or edge points gradients have to be normalized.");
    }

    
    for (int iEdgePoint = 0; iEdgePoint < pointCount; ++iEdgePoint )
    {
        EdgePoint& p = *edgeCollection(iEdgePoint);
        
        // Alternate from the edge point found in the direction opposed to the gradient
        // direction.
        EdgePoint* current = edgeCollection.before(&p);
        // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
        EdgePoint* choosen = nullptr;

        // To save all sub-segments length
        std::vector<float> vDist; ///
        vDist.reserve(params._nCrowns * 2 - 1);

        // Length of the reconstructed field line approximation between the two
        // extremities.
        float totalDistance = 0.f;

        if (current != nullptr)
        {
            // difference in subsequent gradients orientation
            float cosDiffTheta = -p.gradient().dot(current->gradient());
            if (cosDiffTheta >= params._angleVoting)
            {
                float lastDist = cctag::numerical::distancePoints2D(p, *current);
                vDist.push_back(lastDist);
                
                // Add the sub-segment length to the total distance.
                totalDistance += lastDist;

                std::size_t i = 1;
                // Iterate over all crowns
                while (i < params._nCrowns)
                {
                    choosen = nullptr;
                    
                    // First in the gradient direction
                    EdgePoint* target = edgeCollection.after(current);
                    // No edge point was found in that direction
                    if (target == nullptr)
                    {
                        break;
                    }
                    
                    // Check the difference of two consecutive angles
                    cosDiffTheta = -target->gradient().dot(current->gradient());
                    if (cosDiffTheta >= params._angleVoting)
                    {
                        // scalar used to compute the distance ratio
                        float dist = cctag::numerical::distancePoints2D(*target, *current);
                        vDist.push_back(dist);
                        totalDistance += dist;

                        int flagDist = 1;

                        // Check the distance ratio
                        if (vDist.size() > 1)
                        {
                            for (int iDist = 0; iDist < vDist.size(); ++iDist)
                            {
                                for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist)
                                {
                                    flagDist = (vDist[iDist] <= vDist[jDist] * params._ratioVoting) && (vDist[jDist] <= vDist[iDist] * params._ratioVoting) && flagDist;
                                }
                            }
                        }

                        if (flagDist != 0)
                        {
                            lastDist = dist;
                            current = target;
                            // Second in the opposite gradient direction
                            target = edgeCollection.before(current);
                            if (target == nullptr)
                            {
                                break;
                            }
                            cosDiffTheta = -target->gradient().dot(current->gradient());
                            if (cosDiffTheta >= params._angleVoting)
                            {
                                dist = cctag::numerical::distancePoints2D(*target, *current);
                                vDist.push_back(dist);
                                totalDistance += dist;

                                for (int iDist = 0; iDist < vDist.size(); ++iDist)
                                {
                                    for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist)
                                    {
                                        flagDist = (vDist[iDist] <= vDist[jDist] * params._ratioVoting) && (vDist[jDist] <= vDist[iDist] * params._ratioVoting) && flagDist;
                                    }
                                }

                                if (flagDist)
                                {
                                    lastDist = dist;
                                    current = target;
                                    choosen = current;
                                    if (current == nullptr)
                                    {
                                        break;
                                    }
                                }
                                else
                                {
                                    break;
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                    ++i;
                } // while
            }
        }
        // Check if winner was found
        if (choosen != nullptr)
        {
            int iChoosen = edgeCollection(choosen);
            voters[iChoosen].push_back(edgeCollection(&p));
            // update flow length average scale factor
            choosen->_flowLength = (choosen->_flowLength * (voters[iChoosen].size() - 1) + totalDistance) / voters[iChoosen].size();

            // If choosen has a number of votes greater than one of
            // the edge points, then update max.
            if (voters[iChoosen].size() >= params._minVotesToSelectCandidate) {
                if (choosen->_isMax == -1) {
                    seeds.push_back(choosen);
                }
                choosen->_isMax = voters[iChoosen].size();
            }
        }
    }
    edgeCollection.create_voter_lists(voters);
    
    CCTAG_COUT_LILIAN("Elapsed time for vote: " << t.elapsed());
}

    static inline unsigned packxy(int x, int y)
    {
      unsigned ux = x, uy = y;
      return (ux << 16) | (uy & 0xFFFF);
    }

    void edgeLinking(EdgePointCollection& edgeCollection, std::list<EdgePoint*>& convexEdgeSegment, EdgePoint* pmax,
            std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin) {
        
        boost::container::flat_set<unsigned int> processed; // (x,y) packed in 32 bits
        if (pmax) {
            // Add current max point
            convexEdgeSegment.push_back(pmax);
            edgeCollection.set_processed_in(pmax, true);

            processed.insert(packxy(pmax->x(), pmax->y()));
            // Link left
            edgeLinkingDir(edgeCollection, processed, pmax, 1, convexEdgeSegment, windowSizeOnInnerEllipticSegment, averageVoteMin);
            // Link right
            edgeLinkingDir(edgeCollection, processed, pmax, -1, convexEdgeSegment, windowSizeOnInnerEllipticSegment, averageVoteMin);
        }
    }
    
    void edgeLinkingDir(EdgePointCollection& edgeCollection,
                        boost::container::flat_set<unsigned int>& processed,
                        const EdgePoint* p,
                        int dir,
                        std::list<EdgePoint*>& convexEdgeSegment,
                        std::size_t windowSizeOnInnerEllipticSegment,
                        float averageVoteMin) {
        
        std::deque<float> phi;
        std::size_t i = 0;
        bool found = true;

        //int score = p->_isMax;

        int stop = 0;
        std::size_t maxLength = 100;

        float averageVote = edgeCollection.voters_size(p);

        while ((i < maxLength) && (found) && (averageVote >= averageVoteMin) )
        {

            float angle = std::fmod(float(std::atan2(p->dY(), p->dX())) + 2.0f * boost::math::constants::pi<float>(), 2.0f * boost::math::constants::pi<float>());
            phi.push_back(angle);
            //sumPhi += angle;

            if (phi.size() > windowSizeOnInnerEllipticSegment)
            {
                phi.pop_front();

            }

            int shifting = (int) boost::math::round(((angle + boost::math::constants::pi<float>() / 4.0f) / (2.0f * boost::math::constants::pi<float>())) * 8.0f) - 1;

            static int xoff[] = {1, 1, 0, -1, -1, -1, 0, 1};
            int yoff[] = {0, dir * -1, dir * -1, dir * -1, 0, dir * 1, dir * 1, dir * 1};

            int j = 0;
            stop = 0;
            int sx, sy;

            while (!stop) {
                // check high bound
                if (j >= 8) {
                    stop = EDGE_NOT_FOUND; // Not found
                }                                // else if unprocessed point exist
                else {
                    if (dir == 1) {
                        sx = (int) p->x() + xoff[(8 - shifting + j) % 8];
                        sy = (int) p->y() + yoff[(8 - shifting + j) % 8];
                    } else {
                        sx = (int) p->x() + xoff[(shifting + j) % 8];
                        sy = (int) p->y() + yoff[(shifting + j) % 8];
                    }

                    if (sx >= 0 && sx < int( edgeCollection.shape()[0]) &&
                            sy >= 0 && sy < int( edgeCollection.shape()[1]) &&
                            edgeCollection(sx,sy) && processed.find(packxy(sx, sy)) == processed.end()) {
                        if (phi.size() == windowSizeOnInnerEllipticSegment)
                        {
                            // Check if convexity has been lost (concavity)
                            if (dir * std::sin(phi.back() - phi.front()) < 0.0f) {
                                stop = CONVEXITY_LOST; // Convexity lost, stop !
                            }
                        } else {
                            float s = dir * std::sin(phi.back() - phi.front());
                            float c = std::cos(phi.back() - phi.front());
                            // Check if convexity has been lost while the windows is not completed
                            if (((s < -0.707f) && (c > 0.f)) || ((s < 0.f) && (c < 0.f)))
                            {
                                stop = CONVEXITY_LOST; // Convexity lost, stop !
                            }
                        }
                        if (stop == 0) {
                            processed.insert(packxy(p->x(), p->y()));

                            p = edgeCollection(sx,sy);
                            if (dir > 0) {
                                convexEdgeSegment.push_back(const_cast<EdgePoint*>(p));
                            } else {
                                convexEdgeSegment.push_front(const_cast<EdgePoint*>(p));
                            }
                            averageVote = (averageVote*convexEdgeSegment.size() + edgeCollection.voters_size(p)) / ( convexEdgeSegment.size()+1.f );
                            stop = 1; // Found

                        }
                        processed.insert(packxy(p->x(), p->y()));
                    }
                }
                ++j;
            }
            found = stop == 1;
            ++i;
        }

        if ((i == maxLength) || (stop == CONVEXITY_LOST))
        {
            if (convexEdgeSegment.size() > windowSizeOnInnerEllipticSegment)
            {
                int n = 0;
                for(EdgePoint * collectedP : convexEdgeSegment)
                {
                    if (n == convexEdgeSegment.size() - windowSizeOnInnerEllipticSegment)
                    {
                        break;
                    }
                    else
                    {
                        edgeCollection.set_processed_in(collectedP, true);
                        ++n;
                    }
                }
            }
        }
        else if (stop == EDGE_NOT_FOUND)
        {
            for (EdgePoint* collectedP : convexEdgeSegment)
            {
                edgeCollection.set_processed_in(collectedP, true);
            }
        }
        return;
    }

    void childrenOf(const EdgePointCollection& edgeCollection, const std::list<EdgePoint*>& edges, std::list<EdgePoint*>& children) {
        std::size_t voteMax = 1;

        for (const EdgePoint* e : edges) {
          voteMax = std::max((int)voteMax, edgeCollection.voters_size(e));
        }

        for (const EdgePoint* e: edges) {
            auto voters = edgeCollection.voters(e);
            if (voters.second > voters.first) {

                if (voters.second - voters.first >= voteMax / 14)
                for (; voters.first != voters.second; ++voters.first)
                  children.push_back(edgeCollection(*voters.first));
            }
        }
    }

    void outlierRemoval(
            const std::list<EdgePoint*>& children,
            std::vector<EdgePoint*>& filteredChildren,
            float & SmFinal,
            float threshold,
            std::size_t weightedType,
            std::size_t maxSize)
    {
      
      filteredChildren.reserve(children.size());
      
      const std::size_t nSubsampleSize = std::min(children.size(), maxSize);
      const float step = (float) children.size()/ (float) nSubsampleSize;

        // Precondition
        if (nSubsampleSize >= 5) {
            numerical::geometry::Ellipse qm;
            float Sm = 10000000.0;
            const float f = 1.f;

            std::vector<Eigen::Vector3f> pts;
            pts.reserve(children.size());

            std::vector<float> weights;
            weights.reserve(children.size());

            // Store a subset of EdgePoint* on which the robust ellipse estimation will
            // performed. Considering a subset of points is valid as what matters is the 
            // ratio nbInliers/nbOutliers/nbInliers which is irrespective of the size of
            // the set of points to process. 
            // The sub sampled have to be regularly sampled though: todo

            std::size_t k = 0;
            std::size_t iEdgePoint = 0;
            for(const auto edgePoint : children )
            {
              if (iEdgePoint == std::size_t(k*step) )
              {
                ++k;
                pts.emplace_back(edgePoint->cast<float>());
                CCTagVisualDebug::instance().drawPoint(cctag::Point2d<Eigen::Vector3f>(pts.back()), cctag::color_red);

                if (weightedType == INV_GRAD_WEIGHT) {
                  weights.push_back(255 / (edgePoint->normGradient()));
                }
                
                if (weightedType == INV_SQUARE_GRAD_WEIGHT) {
                  weights.push_back(255 / (edgePoint->normGradient())*(edgePoint->normGradient()));
                }
                
              }
              ++iEdgePoint;
            }
            
            Eigen::Matrix<float, 5, 5> A;
            Eigen::Matrix<float, 5, 1> b, temp;

            std::size_t counter = 0;
            std::array<int, 5> perm;
            while (counter < 70)
            {
                // Random subset of 5 points from pts
                //const std::vector<int> perm = cctag::numerical::randperm< std::vector<int> >(pts.size());
                cctag::numerical::rand_5_k(perm, pts.size());
                A.fill(0.f);

                for (std::size_t i = 0; i < 5; ++i) {
                    A(i, 0) = pts[perm[i]](0) * pts[perm[i]](0);
                    A(i, 1) = 2.0f * pts[perm[i]](0) * pts[perm[i]](1);
                    A(i, 2) = pts[perm[i]](1) * pts[perm[i]](1);
                    A(i, 3) = 2.0f * f * pts[perm[i]](0);
                    A(i, 4) = 2.0f * f * pts[perm[i]](1);

                    b(i) = -f * f;
                }
                
                // If A is invertible
                if (A.determinant() != 0.f) {
                    // With PartialPivLU, A MUST be square and invertible. Speed: ++
                    temp = A.lu().solve(b);

                    // Is the conic encoded in temp an ellipse ?
                    if (temp(0) * temp(2) - temp(1) * temp(1) > 0) {
                        Eigen::Matrix3f Q = Eigen::Matrix3f::Zero();
                        Q(0, 0) = temp(0);
                        Q(0, 1) = Q(1, 0) = temp(1);
                        Q(0, 2) = Q(2, 0) = temp(3);
                        Q(1, 1) = temp(2);
                        Q(1, 2) = Q(2, 1) = temp(4);
                        Q(2, 2) = 1.f;

                        try {

                            numerical::geometry::Ellipse q(Q);

                            // Degenerate case ?
                            float ratioSemiAxes = q.a() / q.b();

                            if ((ratioSemiAxes < 0.04f) || (ratioSemiAxes > 25)) {
                                ++counter;
                                continue;
                            }

                            // Compute the median from the set of points pts
                            std::vector<float> dist;
                            numerical::distancePointEllipse(dist, pts, q);

                            if (weightedType != NO_WEIGHT) // todo
                            {
                                for (int iDist = 0; iDist < dist.size(); ++iDist) {
                                    dist[iDist] = dist[iDist] * weights[iDist];
                                }
                            }

                            const float S = numerical::medianRef(dist);

                            if (S < Sm) {
                                counter = 0;
                                qm = numerical::geometry::Ellipse(Q);
                                Sm = S;
                            } else {
                                ++counter;
                            }
                            }catch( ... ){
                        }
                    } else {
                        ++counter;
                    }
                } else {
                    ++counter;
                }
            }

            std::vector<float> vDistFinal;
            vDistFinal.clear();
            vDistFinal.reserve(children.size());

            for(EdgePoint * e : children) {

                float distFinal = std::numeric_limits<float>::max();

                if (weightedType == NO_WEIGHT) {
                  distFinal = numerical::distancePointEllipse(*e, qm);
                } else if (weightedType == INV_GRAD_WEIGHT) {
                  distFinal = numerical::distancePointEllipse(*e, qm)*255 / (e->normGradient());
                } else if (weightedType == INV_SQUARE_GRAD_WEIGHT) {
                  distFinal = numerical::distancePointEllipse(*e, qm)*255 / ((e->normGradient())*(e->normGradient()));
                }
 
                if (distFinal < threshold * Sm) {

                    filteredChildren.push_back(e);
                    vDistFinal.push_back(distFinal);
                }
            }
            if (vDistFinal.empty())
            {
		// Return without modification of the output SmFinal
                return;
            }
            SmFinal = numerical::medianRef(vDistFinal);
        }
    }

    bool isAnotherSegment(
            EdgePointCollection& edgeCollection,
            numerical::geometry::Ellipse & outerEllipse,
            std::vector<EdgePoint*>& outerEllipsePoints,
            const std::vector<EdgePoint*>& filteredChildren,
            const Candidate & anotherCandidate,
            std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > >& cctagPoints,
            std::size_t numCircles,
            float thrMedianDistanceEllipse)
    {
        const std::vector<EdgePoint*> & anotherOuterEllipsePoints = anotherCandidate._outerEllipsePoints;

        numerical::geometry::Ellipse qm;
        float Sm = std::numeric_limits<float>::max();
        const float f = 1.f;

        // Copy/Align content of outerEllipsePoints
        std::vector<Eigen::Vector3f> pts;
        pts.reserve(outerEllipsePoints.size());
        for(const auto & outerEllipsePoint : outerEllipsePoints)
        {
            pts.push_back(outerEllipsePoint->cast<float>());
        }

        // Copy/Align content of anotherOuterEllipsePoints
        std::vector<Eigen::Vector3f> anotherPts;
        anotherPts.reserve(anotherOuterEllipsePoints.size());
        for(const auto & anotherOuterEllipsePoint : anotherOuterEllipsePoints)
        {
            // todo@Lilian: avoid copy, idem in outlierRemoval
            anotherPts.push_back(anotherOuterEllipsePoint->cast<float>());
        }

        std::vector<float> distRef;
        numerical::distancePointEllipse(distRef, pts, outerEllipse);

        const float SRef = numerical::medianRef(distRef);

        std::size_t cnt = 0;

        std::array<int, 5> permutations;
        std::vector<cctag::Point2d<Eigen::Vector3f> > points;
        points.reserve(5);
        while (cnt < 100)
        {
            points.clear(); // Capacity is kept, but the elements are all erased
            // Random subset of 5 points from pts
            cctag::numerical::rand_5_k(permutations, outerEllipsePoints.size());
            
            auto it = permutations.begin();
            for (size_t i = 0; i < 4; ++i) {
                points.emplace_back(float(pts[*it](0)), float(pts[*it](1)));
                ++it;
            }

            cctag::numerical::rand_5_k(permutations, anotherOuterEllipsePoints.size());

            it = permutations.begin();
            for (size_t i = 0; i < 4; ++i) {
                points.emplace_back(float(anotherOuterEllipsePoints[*it]->x()), float(anotherOuterEllipsePoints[*it]->y()));
                ++it;
            }

            if( points.size() < 5 ) {
                std::cerr << __FILE__ << ":" << __LINE__ << " not enough points for fitEllipse" << std::endl;
                continue;
            }

            numerical::geometry::Ellipse eToto;
            cctag::numerical::geometry::fitEllipse(points, eToto);

            try {
                
                numerical::geometry::Ellipse q(eToto.matrix());

                float ratioSemiAxes = q.a() / q.b();

                if ((ratioSemiAxes < 0.12) || (ratioSemiAxes > 8)) {
                    ++cnt;
                    continue;
                }

                std::vector<float> dist;
                numerical::distancePointEllipse(dist, pts, q);

                const float S1 = numerical::medianRef(dist);

                std::vector<float> anotherDist;

                numerical::distancePointEllipse(anotherDist, anotherPts, q);

                const float S2 = numerical::medianRef(anotherDist);

                const float S = S1 + S2;

                if (S < Sm) {
                    cnt = 0;
                    qm = q;
                    Sm = S;

                } else {
                    ++cnt;
                }
            } catch (...) {
                ++cnt;
            }
        }

        float thr = 6;

        if (Sm < thr * SRef) {

            CCTAG_COUT_DEBUG("Median test succed !!\n");

            CCTAG_COUT_VAR_DEBUG(outerEllipse.matrix());
            CCTAG_COUT_VAR_DEBUG(outerEllipsePoints.size());

            std::vector<EdgePoint*> outerEllipsePointsTemp = outerEllipsePoints;
            numerical::geometry::Ellipse outerEllipseTemp;

            outerEllipsePointsTemp.insert(outerEllipsePointsTemp.end(), anotherOuterEllipsePoints.begin(), anotherOuterEllipsePoints.end());
            CCTAG_COUT_VAR_DEBUG(outerEllipsePointsTemp.size());
            
            // Compute the new ellipse which fits oulierEllipsePoints
            numerical::ellipseFitting(outerEllipseTemp, outerEllipsePointsTemp);

            float quality = (float) outerEllipsePointsTemp.size() / (float) rasterizeEllipsePerimeter(outerEllipseTemp);

            if (quality < 1.1) {
                std::vector<float> vDistFinal;
                vDistFinal.reserve(outerEllipsePointsTemp.size());

                for(EdgePoint * p : outerEllipsePointsTemp) {
                    float distFinal = numerical::distancePointEllipse(*p, outerEllipseTemp);
                    vDistFinal.push_back(distFinal);
                }
                const float SmFinal = numerical::medianRef(vDistFinal);

                if (SmFinal < thrMedianDistanceEllipse) {
                    if (addCandidateFlowtoCCTag(edgeCollection, anotherCandidate._filteredChildren, anotherOuterEllipsePoints, outerEllipseTemp, cctagPoints, numCircles)) {
                        outerEllipsePoints = outerEllipsePointsTemp;
                        outerEllipse = outerEllipseTemp;

                        CCTAG_COUT_VAR_DEBUG(outerEllipse);
                        return true;
                    } else {
                        CCTAG_COUT_DEBUG("isAnotherSegment : intermediate points outside or bad gradient orientations: ");
                        return false;
                    }
                } else {
                    CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(FINAL_MEDIAN_TEST_FAILED_WHILE_ASSEMBLING);
                    CCTAG_COUT_DEBUG("SmFinal > thrMedianDistanceEllipse in isAnotherSegment");
                }
            } else {
                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(QUALITY_TEST_FAILED_WHILE_ASSEMBLING);
                CCTAG_COUT_DEBUG("Quality too high: " << quality);
                return false;
            }
        } else {
            CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(MEDIAN_TEST_FAILED_WHILE_ASSEMBLING);
            CCTAG_COUT_DEBUG("Test failed !!\n");
            return false;
        }
        return false;
    }
            
} // namespace cctag


