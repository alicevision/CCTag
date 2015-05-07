#define BOOST_UBLAS_TYPE_CHECK 0

#include "brensenham.hpp"
#include "EdgePoint.hpp"
#include "types.hpp"
#include "vote.hpp"
#include "toolbox.hpp"
#include "ellipse.hpp"
#include "fileDebug.hpp"

#include <cctag/geometry/point.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/global.hpp>

#include <boost/foreach.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>

#include <boost/format/format_implementation.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/multi_array/multi_array_ref.hpp>
#include <boost/multi_array/subarray.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/unordered/unordered_map.hpp>

#include <algorithm>
#include <cmath>
#include <ostream>

#define EDGE_NOT_FOUND -1
#define CONVEXITY_LOST -2
#define LOW_FLOW -3

namespace cctag {
    namespace vision {
        namespace marker {

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
 * cannyGradX: X derivative of the gray image
 * cannyGradY: Y derivative of the gray image
 */
void vote(std::vector<EdgePoint> & points, std::vector<EdgePoint*> & seeds,
        const EdgePointsImage & edgesMap, WinnerMap& winners,
        const boost::gil::kth_channel_view_type<1,boost::gil::rgb32f_view_t>::type & cannyGradX,
        const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
        const Parameters & params)
{
    BOOST_FOREACH(EdgePoint & p, points) {
        p._before = gradientDirectionDescent(edgesMap, p, -1, params._distSearch, cannyGradX, cannyGradY, params._thrGradientMagInVote);
        p._after = gradientDirectionDescent(edgesMap, p, 1, params._distSearch, cannyGradX, cannyGradY, params._thrGradientMagInVote);
    }
    // Vote
    seeds.reserve(points.size() / 2);

    // todo@Lilian: remove thrVotingAngle from the paramter file
    if (params._angleVoting != 0) {
        BOOST_THROW_EXCEPTION(cctag::exception::Bug() << cctag::exception::user() + 
                "thrVotingAngle must be equal to 0 or edge points gradients have to be normalized");
    }

    BOOST_FOREACH(EdgePoint & p, points) {
        float lastDist, dist, totalDistance; // scalar to compute the distance ratio
        float cosDiffTheta; // difference in subsequent gradients orientation
        std::size_t i = 1;
        
        // Alternate from the edge point found in the direction opposed to the gradient
        // direction.
        EdgePoint* current = p._before;
        // Here current contains the edge point lying on the 2nd ellipse (from outer to inner)
        EdgePoint* choosen = NULL;

        // To save all sub-segments length
        std::vector<float> vDist; ///
        vDist.reserve(params._numCrowns * 2 - 1);
        int flagDist = 1;

        // Length of the reconstructed field line approximation between the two
        // extremities.
        totalDistance = 0.0;

        if (current) {
            cosDiffTheta = -inner_prod(subrange(p._grad, 0, 2), subrange(current->_grad, 0, 2));
            if (cosDiffTheta >= params._angleVoting) {
                lastDist = cctag::numerical::distancePoints2D(p, *current);
                vDist.push_back(lastDist);
                
                // Add the sub-segment length to the total distance.
                totalDistance += lastDist;

                // Iterate over all crowns
                while (i < params._numCrowns) {
                    choosen = NULL;
                    
                    // First in the gradient direction
                    EdgePoint* target = current->_after;
                    // No edge point was found in that direction
                    if (!target) {
                        break;
                    }
                    
                    // Check the difference of two consecutive angles
                    cosDiffTheta = -inner_prod(subrange(target->_grad, 0, 2), subrange(current->_grad, 0, 2));
                    if (cosDiffTheta >= params._angleVoting) {
                        dist = cctag::numerical::distancePoints2D(*target, *current);
                        vDist.push_back(dist);
                        totalDistance += dist;

                        // Check the distance ratio
                        if (vDist.size() > 1) {
                            for (int iDist = 0; iDist < vDist.size(); ++iDist) {
                                for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist) {
                                    flagDist = (vDist[iDist] <= vDist[jDist] * params._ratioVoting) && (vDist[jDist] <= vDist[iDist] * params._ratioVoting) && flagDist;
                                }
                            }
                        }

                        if (flagDist)
                        {
                            lastDist = dist;
                            current = target;
                            // Second in the opposite gradient direction
                            target = current->_before;
                            if (!target) {
                                break;
                            }
                            cosDiffTheta = -inner_prod(subrange(target->_grad, 0, 2), subrange(current->_grad, 0, 2));
                            if (cosDiffTheta >= params._angleVoting) {
                                dist = cctag::numerical::distancePoints2D(*target, *current);
                                vDist.push_back(dist);
                                totalDistance += dist;

                                for (int iDist = 0; iDist < vDist.size(); ++iDist) {
                                    for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist) {
                                        flagDist = (vDist[iDist] <= vDist[jDist] * params._ratioVoting) && (vDist[jDist] <= vDist[iDist] * params._ratioVoting) && flagDist;
                                    }
                                }

                                if (flagDist)
                                {
                                    lastDist = dist;
                                    current = target;
                                    choosen = current;
                                    if (!current) {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                    ++i;
                }
            }
        }
        // Check if winner was found
        if (choosen) {
            // Associate winner with its voter (add the current point)
            winners[choosen].push_back(&p);

            // update flow length average scale factor
            choosen->_flowLength = (choosen->_flowLength * (winners[choosen].size() - 1) + totalDistance) / winners[choosen].size();

            // If choosen has a number of votes greater than one of
            // the edge points, then update max.
            if (winners[choosen].size() >= params._minVotesToSelectCandidate) {
                if (choosen->_isMax == -1) {
                    seeds.push_back(choosen);
                }
                choosen->_isMax = winners[choosen].size();
            }
        }
    }
    CCTAG_COUT_LILIAN("Elapsed time for vote: " << t.elapsed());
}

    void edgeLinking(const EdgePointsImage& img, std::list<EdgePoint*>& convexEdgeSegment, EdgePoint* pmax,
            WinnerMap& winners, std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin) {
        boost::unordered_set< std::pair<int, int> > processed;
        if (pmax) {
            // Add current max point
            convexEdgeSegment.push_back(pmax);
            pmax->_processedIn = true;

            processed.insert(std::pair < int, int > ((int) pmax->x(), (int) pmax->y()));
            // Link left
            edgeLinkingDir(img, processed, pmax, 1, convexEdgeSegment, winners, windowSizeOnInnerEllipticSegment, averageVoteMin);
            // Link right
            edgeLinkingDir(img, processed, pmax, -1, convexEdgeSegment, winners, windowSizeOnInnerEllipticSegment, averageVoteMin);
        }
    }

    void edgeLinkingDir(const EdgePointsImage& img, boost::unordered_set< std::pair<int, int> >& processed, EdgePoint* p, const int dir,
            std::list<EdgePoint*>& convexEdgeSegment, WinnerMap& winners, std::size_t windowSizeOnInnerEllipticSegment, float averageVoteMin) {
        std::list<float> phi;
        std::size_t i = 0;
        bool found = true;

        //int score = p->_isMax;

        int stop = 0;
        std::size_t maxLength = 100;

        float averageVote = winners[p].size();

        while ((i < maxLength) && (found) && (averageVote >= averageVoteMin) )
        {

            float angle = std::fmod(float(std::atan2(p->_grad.y(), p->_grad.x())) + 2.0f * boost::math::constants::pi<float>(), 2.0f * boost::math::constants::pi<float>());
            phi.push_back(angle);
            //sumPhi += angle;

            if (phi.size() > windowSizeOnInnerEllipticSegment) // TODO , 4 est un paramètre de l'algorithme, + les motifs à détecter sont importants, + la taille de la fenêtre doit être grande
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
                        sx = (int) p->x() + xoff[(8 - shifting + j) % 8]; //int sx = p->x() + xoff[(j + shifting) % 8];
                        sy = (int) p->y() + yoff[(8 - shifting + j) % 8]; //int sy = p->y() + yoff[(j + shifting) % 8];
                    } else {
                        sx = (int) p->x() + xoff[(shifting + j) % 8]; //int sx = p->x() + xoff[(j + shifting) % 8];
                        sy = (int) p->y() + yoff[(shifting + j) % 8];
                    }

                    if (sx >= 0 && sx < int( img.shape()[0]) &&
                            sy >= 0 && sy < int( img.shape()[1]) &&
                            img[sx][sy] && processed.find(std::pair<int, int>(sx, sy)) == processed.end()) {
                        if (phi.size() == windowSizeOnInnerEllipticSegment) // (ok, resolu avec la multiresolution) TODO , 4 est un paramètre de l'algorithme, + les motifs à détecter sont importants, + la taille de la fenêtre doit être grande
                        {
                            // Check if convexity has been lost (concavity)
                            if (dir * std::sin(phi.back() - phi.front()) < 0.0f) {
                                stop = CONVEXITY_LOST; // Convexity lost, stop !
                            }
                        } else {
                            float s = dir * std::sin(phi.back() - phi.front());
                            float c = std::cos(phi.back() - phi.front());
                            // Check if convexity has been lost (concavity) while the windows is not completed -- do smthing like this for the previous test todo@Lilian
                            if (((s < -0.707) && (c > 0)) || ((s < 0) && (c < 0)))// stop pour un angle > pi/4
                            {
                                stop = CONVEXITY_LOST; // Convexity lost, stop !
                            }
                        }
                        if (stop == 0) {
                            processed.insert(std::pair < int, int > ((int) p->x(), (int) p->y()));
                            /* if no thinning, uncomment this part */
                            ///////////////Bloc lié à la différence de réponse du détecteur de contour//////////////////////
                            /*{
                                int ind;
                                if (j<=2)
                                {
                                    if (dir==1)
                                    {
                                        ind = (8 - shifting + j + 1) % 8;
                                        if ((ind==0)||(ind==2)||(ind==4)||(ind==6))
                                        {
                                            int sx2 = (int)p->x() + xoff[ind];
                                            int sy2 = (int)p->y() + yoff[ind];

                                            if ( sx2 >= 0 && sx2 < img.shape()[0] &&
                                                     sy2 >= 0 && sy2 < img.shape()[1] &&
                                                     img[sx2][sy2] && processed.find(std::pair<int, int>(sx2, sy2)) == processed.end() )
                                            {
                                                processed[std::pair<int, int>(sx2, sy2)] = true;
                                                convexEdgeSegment.push_back(img[sx2][sy2]);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        ind = (shifting + j + 1) % 8;
                                        if ((ind==0)||(ind==2)||(ind==4)||(ind==6))
                                        {
                                            int sx2 = (int)p->x() + xoff[ind];
                                            int sy2 = (int)p->y() + yoff[ind];
                                            if ( sx2 >= 0 && sx2 < img.shape()[0] &&
                                                     sy2 >= 0 && sy2 < img.shape()[1] &&
                                                     img[sx2][sy2] && processed.find(std::pair<int, int>(sx2, sy2)) == processed.end() )
                                            {
                                                processed[std::pair<int, int>(sx2, sy2)] = true;
                                                convexEdgeSegment.push_front(img[sx2][sy2]);
                                            }
                                        }
                                    }
                                }
                               }*/
                            //////////////////////////////////fin bloc////////////////////////////////
                            p = img[sx][sy];
                            if (dir > 0) {
                                convexEdgeSegment.push_back(p);
                            } else {
                                convexEdgeSegment.push_front(p);
                            }
                            averageVote = (averageVote*convexEdgeSegment.size() + winners[p].size() )/ ( convexEdgeSegment.size()+1.0 );
                            stop = 1; // Found

                        }
                        processed.insert(std::pair < int, int > ((int) p->x(), (int) p->y()));
                    }
                }
                ++j;
            }
            found = stop == 1;
            ++i;
        }

        int n = 0;
        if ((i == maxLength) || (stop == CONVEXITY_LOST)) {
            if (convexEdgeSegment.size() > windowSizeOnInnerEllipticSegment) {
                BOOST_FOREACH(EdgePoint * collectedP, convexEdgeSegment) {
                    if (n == convexEdgeSegment.size() - windowSizeOnInnerEllipticSegment) {
                        break;
                    } else {
                        collectedP->_processedIn = true;
                        ++n;
                    }
                }
            }
        } else if (stop == EDGE_NOT_FOUND) {

            BOOST_FOREACH(EdgePoint* collectedP, convexEdgeSegment) {
                collectedP->_processedIn = true;
            }
        }
        return;
    }

    void childrensOf(std::list<EdgePoint*>& edges, WinnerMap& winnerMap, std::list<EdgePoint*>& childrens) {
        std::size_t voteMax = 1;

        // OPTI@Lilian : the maximum vote can be computed in the edge linking step with low cost.

        BOOST_FOREACH(EdgePoint * e, edges) {
            voteMax = std::max(voteMax, winnerMap[e].size());
        }

        BOOST_FOREACH(EdgePoint * e, edges) {
            if (winnerMap[e].size()) {
                //childrens.splice( childrens.end(), winnerMap[e] );

                if (winnerMap[e].size() >= voteMax / 14) // keep outer ellipse point associated with small curvature ! ( near to the osculting circle).  delete this line @Lilian ?
                {
                    childrens.insert(childrens.end(), winnerMap[e].begin(), winnerMap[e].end());
                }
            }
        }
    }

    void outlierRemoval(const std::list<EdgePoint*>& childrens, std::vector<EdgePoint*>& filteredChildrens, double & SmFinal, double threshold, std::size_t weightedType) {
        using namespace boost::numeric::ublas;
        // function [Qm, Sm, pts_in, param_in,flag,i_inliers] = outlierRemoval(pts,debug)
        // outLierRemoval compute from a set of points pts the best ellipse which fits
        // a subset of pts i.e. with Sm minimal

        //std::srand(1);

        filteredChildrens.reserve(childrens.size());

        const std::size_t n = childrens.size();
        // Precondition
        if (n >= 5) {
            numerical::geometry::Ellipse qm;
            double Sm = 10000000.0;
            const double f = 1.0;

            // TODO, le passage en coordonnées homogène pas nécessaire, il faut alors modifier
            // la résolution du système linéaire ci-dessous

            // Create matrix containing homogeneous coordinates of childrens

            std::vector<bounded_vector<double, 3> > pts;
            pts.reserve(childrens.size());

            std::vector<double> weights;
            weights.reserve(childrens.size());

            for (std::list<EdgePoint*>::const_iterator it = childrens.begin(); it != childrens.end(); ++it) {
                // Push_back...
                // TODO chaque point doit être x/w y/w 1: note eloi: utilise la methode cartesian() de Point2dN
                pts.push_back(*(*it));

                if (weightedType == INV_GRAD_WEIGHT) {
                    weights.push_back(255 / ((*it)->_normGrad)); //  0.003921= 1/255
                }
            }

            std::size_t cnt = 0;

            while (cnt < 70)
            {
                // Random subset of 5 points from pts
                const std::vector<int> perm = cctag::numerical::randperm< std::vector<int> >(n);

                bounded_matrix<double, 5, 5> A(5, 5);
                bounded_vector<double, 5> b(5);

                // TODO : Résoltion du pb linéaire ou cvFitEllipse ?

                for (std::size_t i = 0; i < 5; ++i) {
                    A(i, 0) = pts[perm[i]](0) * pts[perm[i]](0);
                    A(i, 1) = 2.0 * pts[perm[i]](0) * pts[perm[i]](1);
                    A(i, 2) = pts[perm[i]](1) * pts[perm[i]](1);
                    A(i, 3) = 2.0 * f * pts[perm[i]](0);
                    A(i, 4) = 2.0 * f * pts[perm[i]](1);

                    b(i) = -f * f;
                }

                bounded_matrix<double, 5, 5> AInv(5, 5);
                ///@todo what shall we do when invert fails ?
                if (cctag::numerical::invert(A, AInv)) {
                    bounded_vector<double, 5> temp(5);

                    temp = prec_prod(AInv, b); // prec_prod ou prod ?

                    //Ellipse(const bounded_matrix<double, 3, 3> & matrix)
                    //param = (inv(A)*(-f*f*ones(5,1)))';

                    // The conic encoded in temp is an ellipse ?
                    if (temp(0) * temp(2) - temp(1) * temp(1) > 0) {
                        bounded_matrix<double, 3, 3> Q;
                        Q.clear();
                        Q(0, 0) = temp(0);
                        Q(0, 1) = Q(1, 0) = temp(1);
                        Q(0, 2) = Q(2, 0) = temp(3);
                        Q(1, 1) = temp(2);
                        Q(1, 2) = Q(2, 1) = temp(4);
                        Q(2, 2) = 1.0;

                        try {

                            numerical::geometry::Ellipse q(Q);

                            // Provisoire, cas dégénéré : l'ellipse est 2 droite, cas ou 3 ou 4 points sont alignés sur les 5
                            //bounded_vector<double, 5> paramQ;
                            //q.computeParams(paramQ);
                            double ratioSemiAxes = q.a() / q.b();


                            // if "search for another convex segment" is disabled, uncomment this bloc -- todo@Lilian
                            if ((ratioSemiAxes < 0.04) || (ratioSemiAxes > 25)) {
                                ++cnt;
                                continue;
                            }

                            // We compute the median from the set of points pts

                            //% distance between pts and Q : (xa'Qxa)²/||PkQxa||² with Pk = diag(1,1,0)
                            std::vector<double> dist;
                            numerical::distancePointEllipse(dist, pts, q, f);

                            if (weightedType != NO_WEIGHT) // todo...
                            {
                                for (int iDist = 0; iDist < dist.size(); ++iDist) {
                                    dist[iDist] = dist[iDist] * weights[iDist];
                                }
                            }

                            const double S = numerical::medianRef(dist);

                            if (S < Sm) {
                                cnt = 0;
                                qm = numerical::geometry::Ellipse(Q);
                                Sm = S;
                                //CCTAG_COUT(Sm);
                            } else {
                                ++cnt;
                            }
                            }catch( ... ){
                            // Conique != ellipse, on passe à la une nouvelle permutation aléatoire.
                        }
                    } else {
                        ++cnt;
                    }
                } else {
                    ++cnt;
                }
            }

            std::vector<double> vDistFinal;
            vDistFinal.clear();
            vDistFinal.reserve(childrens.size());

            //CCTAG_COUT_VAR(qm);

            BOOST_FOREACH(EdgePoint * e, childrens) {

                double distFinal;

                if (weightedType == NO_WEIGHT) {
                    distFinal = numerical::distancePointEllipse(*e, qm, f);
                } else if (weightedType == INV_GRAD_WEIGHT) {
                    distFinal = numerical::distancePointEllipse(*e, qm, f)*255 / (e->_normGrad);
                }
 
                if (distFinal < threshold * Sm) {

                    filteredChildrens.push_back(e);
                    vDistFinal.push_back(distFinal);
                }
            }
            
            SmFinal = numerical::medianRef(vDistFinal);
        }
    }

    bool isAnotherSegment(numerical::geometry::Ellipse & outerEllipse, std::vector<EdgePoint*>& outerEllipsePoints, const std::vector<EdgePoint*>& filteredChildrens, const Candidate & anotherCandidate, std::vector< std::vector< Point2dN<double> > >& cctagPoints, std::size_t numCircles, double thrMedianDistanceEllipse) {

        using namespace boost::numeric::ublas;

        const std::vector<EdgePoint*> & anotherOuterEllipsePoints = anotherCandidate._outerEllipsePoints;

        numerical::geometry::Ellipse qm;
        double Sm = 10000000000000.0;
        const double f = 1.0;

        std::vector<bounded_vector<double, 3> > pts;
        pts.reserve(outerEllipsePoints.size());
        for (std::vector<EdgePoint*>::iterator it = outerEllipsePoints.begin(); it != outerEllipsePoints.end(); ++it) {
            // on fait des recopies !! todo@Lilian, idem dans outlierRemoval -- degeulasse --
            pts.push_back(*(*it));
        }


        std::vector<bounded_vector<double, 3> > anotherPts;
        anotherPts.reserve(anotherOuterEllipsePoints.size());
        for (std::vector<EdgePoint*>::const_iterator it = anotherOuterEllipsePoints.begin(); it != anotherOuterEllipsePoints.end(); ++it) {
            // on fait des recopies !! todo@Lilian, idem dans outlierRemoval -- degeulasse --
            anotherPts.push_back(*(*it));
        }

        // distance between pts and Q : (xa'Qxa)²/||PkQxa||² with Pk = diag(1,1,0)
        std::vector<double> distRef;
        numerical::distancePointEllipse(distRef, pts, outerEllipse, f);

        const double SRef = numerical::medianRef(distRef);

        std::size_t cnt = 0;

        double S1m, S2m;

        while (cnt < 100)
        {
            // Random subset of 5 points from pts
            const std::vector<int> perm = cctag::numerical::randperm< std::vector<int> >(outerEllipsePoints.size());

            std::vector<cctag::Point2dN< double > > points;

            std::vector<int>::const_iterator it = perm.begin();
            for (std::size_t i = 0; i < 4; ++i) {
                points.push_back(Point2dN<double>(double(pts[*it](0)), double(pts[*it](1))));
                ++it;
            }

            const std::vector<int> anotherPerm = cctag::numerical::randperm< std::vector<int> >(anotherOuterEllipsePoints.size());

            it = anotherPerm.begin();
            for (std::size_t i = 0; i < 4; ++i) {
                points.push_back(Point2dN<double>(double(anotherOuterEllipsePoints[*it]->x()), double(anotherOuterEllipsePoints[*it]->y())));
                ++it;
            }

            numerical::geometry::Ellipse eToto;
            cctag::numerical::geometry::fitEllipse(points, eToto);

            try {
                
                numerical::geometry::Ellipse q(eToto.matrix());

                // Provisoire, cas dégénéré : l'ellipse est 2 droite, cas ou 3 ou 4 points sont alignés sur les 5
                //bounded_vector<double, 5> paramQ;
                //q.computeParams(paramQ);
                double ratioSemiAxes = q.a() / q.b();

                if ((ratioSemiAxes < 0.12) || (ratioSemiAxes > 8)) {
                    ++cnt;
                    continue;
                }

                // We compute the median from the set of points pts

                //% distance between pts and Q : (xa'Qxa)²/||PkQxa||² with Pk = diag(1,1,0)
                std::vector<double> dist;
                numerical::distancePointEllipse(dist, pts, q, f);

                //for(int iDist=0 ; iDist < dist.size() ; ++iDist)
                //{
                //	dist[iDist] = dist[iDist]*weights[iDist];
                //}

                const double S1 = numerical::medianRef(dist);

                std::vector<double> anotherDist;

                numerical::distancePointEllipse(anotherDist, anotherPts, q, f);

                const double S2 = numerical::medianRef(anotherDist);

                const double S = S1 + S2;

                if (S < Sm) {
                    cnt = 0;
                    qm = q;
                    Sm = S;

                    S1m = S1;
                    S2m = S2;
                } else {
                    ++cnt;
                }
            } catch (...) {
                ++cnt;
            }
        }

        double thr = 6;

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

            double quality = (double) outerEllipsePointsTemp.size() / (double) rasterizeEllipsePerimeter(outerEllipseTemp);

            if (quality < 1.1) { // replace by 1 and perform an outlier removal after the ellipse growing. todo@Lilian
                std::vector<double> vDistFinal;
                vDistFinal.reserve(outerEllipsePointsTemp.size());

                BOOST_FOREACH(EdgePoint * p, outerEllipsePointsTemp) {
                    double distFinal = numerical::distancePointEllipse(*p, outerEllipseTemp, 1.0);
                    vDistFinal.push_back(distFinal);
                }
                const double SmFinal = numerical::medianRef(vDistFinal);
                //const double thrMedianDistanceEllipse = 3; // todo@Lilian -- utiliser le meme seuil que dans la main loop 1

                if (SmFinal < thrMedianDistanceEllipse) {
                    if (addCandidateFlowtoCCTag(anotherCandidate._filteredChildrens, anotherOuterEllipsePoints, outerEllipseTemp, cctagPoints, numCircles)) {
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
            
        }
    }
}


