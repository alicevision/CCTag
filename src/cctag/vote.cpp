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

namespace rom {
    namespace vision {
        namespace marker {
            namespace cctag {

#define NO_WEIGHT 0
#define INV_GRADIENT_WEIGHT 1
#define INV_SQRT_GRADIENT_WEIGHT 2


                /**
                 * @param[in] algo
                 * @param[out] winners
                 */
                //void vote( std::vector<EdgePoint> & points, std::vector<EdgePoint*> & candidates,
                //	const EdgePointsImage & edgesMap, WinnerMap& winners, const std::size_t searchDistance,
                //	const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns,
                //	const std::size_t minVotesToSelectCandidate )
                //{
                //	// Before & after points computation
                //	BOOST_FOREACH( EdgePoint & p, points )
                //	{
                //		p._before = bresenham( edgesMap, p, -1, searchDistance );
                //		p._after  = bresenham( edgesMap, p,  1, searchDistance );
                //	}
                //	// Vote
                //	candidates.reserve( points.size() / 2 );
                //
                //	BOOST_FOREACH( EdgePoint & p, points )
                //	{
                //		float lastDist, dist;
                //		float cosDiffTheta, cosDiffThetaInitial;                      // Gradient angle differences
                //		std::size_t i = 1;
                //		EdgePoint* current = p._before;
                //		EdgePoint* choosen = NULL;
                //
                //		float dimension = 0;
                //		// Remplacer toutes ces heuristiques par un regle de decision bayesienne sur le vecteur-(nbCircles-1) sur les angles
                //		// et les distances todo@Lilian
                //
                //		if( current )
                //		{
                //			cosDiffTheta = -inner_prod( subrange(p._grad,0,2), subrange(current->_grad,0,2) );
                //			if( cosDiffTheta >= thrVotingAngle )
                //			{
                //				lastDist = rom::numerical::distancePoints2D( p, *current );
                //				dimension += lastDist;
                //				while( i < numCrowns )
                //				{
                //					choosen = NULL;
                //					EdgePoint* target = current->_after;
                //					if( !target )
                //					{
                //						break;
                //					}
                //					cosDiffTheta = -inner_prod( subrange(target->_grad,0,2), subrange(current->_grad,0,2) );
                //					if( cosDiffTheta >= thrVotingAngle )
                //					{
                //						dist = rom::numerical::distancePoints2D( *target, *current );
                //						dimension += dist;
                //						if( ( lastDist <= dist * thrVotingRatio ) && ( dist <= lastDist * thrVotingRatio ) )
                //						{
                //							lastDist = dist;
                //							current  = target;
                //							target   = current->_before;
                //							if( !target )
                //							{
                //								break;
                //							}
                //							cosDiffTheta = -inner_prod( subrange(target->_grad,0,2), subrange(current->_grad,0,2) );
                //							// -- added -- todo@Lilian
                //							// cosDiffThetaInitial = -inner_prod( p._grad.getNormalizedBoundedVec2d(), current->_grad.getNormalizedBoundedVec2d() );
                //							// -- added --
                //							if ( ( cosDiffTheta >= thrVotingAngle ) ) //&& ( cosDiffThetaInitial >= thrVotingAngle ) )
                //							{
                //								dist = rom::numerical::distancePoints2D( *target, *current );
                //								dimension += dist;
                //								
                //								if( ( lastDist <= dist * thrVotingRatio ) && ( dist <= lastDist * thrVotingRatio ) )
                //								{
                //									lastDist = dist;
                //									current  = target;
                //									choosen  = current;
                //									if( !current )
                //									{
                //										break;
                //									}
                //								}
                //								else
                //								{
                //									break;
                //								}
                //							}
                //							else
                //							{
                //								break;
                //							}
                //						}
                //						else
                //						{
                //							break;
                //						}
                //					}
                //					else
                //					{
                //						break;
                //					}
                //					++i;
                //				}
                //			}
                //		}
                //		// Check if winner was found
                //		if( choosen )
                //		{
                //			// Associate winner with its voter (add the current point)
                //			winners[choosen].push_back( &p );
                //			
                //			// If choosen has a number of votes greater than one of
                //			// the edge points, then update max.
                //			if( winners[choosen].size() >= minVotesToSelectCandidate )
                //			{
                //				if (choosen->_isMax == -1){
                //					candidates.push_back( choosen );
                //				}
                //				choosen->_isMax = winners[choosen].size();
                //			}
                //		}
                //	}
                //	ROM_COUT_LILIAN( "Elapsed time for vote: " << t.elapsed() );
                //}

                void vote(std::vector<EdgePoint> & points, std::vector<EdgePoint*> & candidates,
                        const EdgePointsImage & edgesMap, WinnerMap& winners, const std::size_t searchDistance,
                        const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns,
                        const std::size_t minVotesToSelectCandidate, const boost::gil::kth_channel_view_type<1,
                        boost::gil::rgb32f_view_t>::type & cannyGradX, const boost::gil::kth_channel_view_type<2,
                        boost::gil::rgb32f_view_t>::type & cannyGradY) {

                    int thrGradient = 50;
                    int thrGradient2 = thrGradient*thrGradient;

                    // Before & after points computation
                    //int counter = 0;

                    BOOST_FOREACH(EdgePoint & p, points) {
                        //if ( counter % 2 == 0 )
                        //{
                        p._before = gradientDirectionDescent(edgesMap, p, -1, searchDistance, cannyGradX, cannyGradY, thrGradient2);
                        p._after = gradientDirectionDescent(edgesMap, p, 1, searchDistance, cannyGradX, cannyGradY, thrGradient2);
                        //}
                        //counter += 1;
                    }
                    // Vote
                    candidates.reserve(points.size() / 2);

                    // todo@Lilian
                    if (thrVotingAngle != 0) {
                        BOOST_THROW_EXCEPTION(rom::exception::Bug() << rom::exception::user() + "thrVotingAngle must be equal to 0 or edge points gradients have to be normalized");
                    }

                    BOOST_FOREACH(EdgePoint & p, points) {
                        float lastDist, dist, totalDistance;
                        float cosDiffTheta; // Gradient angle differences
                        std::size_t i = 1;
                        EdgePoint* current = p._before;
                        EdgePoint* choosen = NULL;

                        std::vector<float> vDist; ///
                        vDist.reserve(numCrowns * 2 - 1); ///
                        int flagDist = 1;

                        totalDistance = 0.0;

                        if (current) {
                            cosDiffTheta = -inner_prod(subrange(p._grad, 0, 2), subrange(current->_grad, 0, 2));
                            if (cosDiffTheta >= thrVotingAngle) {
                                lastDist = rom::numerical::distancePoints2D(p, *current);
                                vDist.push_back(lastDist);
                                totalDistance += lastDist;

                                while (i < numCrowns) {
                                    choosen = NULL;
                                    EdgePoint* target = current->_after;
                                    if (!target) {
                                        break;
                                    }
                                    cosDiffTheta = -inner_prod(subrange(target->_grad, 0, 2), subrange(current->_grad, 0, 2));
                                    if (cosDiffTheta >= thrVotingAngle) {
                                        dist = rom::numerical::distancePoints2D(*target, *current);
                                        vDist.push_back(dist);
                                        totalDistance += dist;

                                        if (vDist.size() > 1) {
                                            for (int iDist = 0; iDist < vDist.size(); ++iDist) {
                                                for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist) {
                                                    flagDist = (vDist[iDist] <= vDist[jDist] * thrVotingRatio) && (vDist[jDist] <= vDist[iDist] * thrVotingRatio) && flagDist;
                                                }
                                            }
                                        }
                                        //if( ( lastDist <= dist * thrVotingRatio ) && ( dist <= lastDist * thrVotingRatio ) )

                                        if (flagDist) // ( ( lastDist <= dist * thrVotingRatio ) && ( dist <= lastDist * thrVotingRatio ) )
                                        {
                                            lastDist = dist;
                                            current = target;
                                            target = current->_before;
                                            if (!target) {
                                                break;
                                            }
                                            cosDiffTheta = -inner_prod(subrange(target->_grad, 0, 2), subrange(current->_grad, 0, 2));
                                            if (cosDiffTheta >= thrVotingAngle) {
                                                dist = rom::numerical::distancePoints2D(*target, *current);
                                                vDist.push_back(dist);
                                                totalDistance += dist;

                                                for (int iDist = 0; iDist < vDist.size(); ++iDist) {
                                                    for (int jDist = iDist + 1; jDist < vDist.size(); ++jDist) {
                                                        flagDist = (vDist[iDist] <= vDist[jDist] * thrVotingRatio) && (vDist[jDist] <= vDist[iDist] * thrVotingRatio) && flagDist;
                                                    }
                                                }

                                                if (flagDist) //( ( lastDist <= dist * thrVotingRatio ) && ( dist <= lastDist * thrVotingRatio ) )
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

                            /*if (( choosen->x() == 520 ) && ( choosen->y() == 172) )
                            {
                                    ROM_COUT_VAR(totalDistance);
                                    ROM_COUT_VAR(choosen->_flowLength);
                                    std::cin.ignore().get();
                            }*/

                            // If choosen has a number of votes greater than one of
                            // the edge points, then update max.
                            if (winners[choosen].size() >= minVotesToSelectCandidate) {
                                if (choosen->_isMax == -1) {
                                    candidates.push_back(choosen);

                                    //for(int i=1 ; i<numCrowns-1 ; ++i){
                                    //	choosen->_vfieldLineRatio[i] = vDist[i]/vDist[i-1];
                                    //}
                                } else {
                                    //for(int i=1 ; i<numCrowns-1 ; ++i){
                                    //	choosen->_vfieldLineRatio[i] = ( choosen->_vfieldLineRatio[i]*(winners[choosen].size()-1) + vDist[i]/vDist[i-1] )/winners[choosen].size();
                                    //}
                                }
                                choosen->_isMax = winners[choosen].size();

                                /*if ((choosen->x() == 331) && (choosen->y() == 209)){
                                        ROM_COUT_VAR(totalDistance);
                                        ROM_COUT_VAR((choosen->_flowLength*(winners[choosen].size()-1) + totalDistance)/winners[choosen].size());
                                        std::cin.ignore().get();
                                }*/
                            }
                        }
                    }
                    ROM_COUT_LILIAN("Elapsed time for vote: " << t.elapsed());
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

                            if (weightedType == INV_GRADIENT_WEIGHT) {
                                weights.push_back(255 / ((*it)->_normGrad)); //  0.003921= 1/255
                            }
                        }

                        std::size_t cnt = 0;

                        while (cnt < 70)//50  ////////////
                        {
                            // Random subset of 5 points from pts
                            const std::vector<int> perm = rom::numerical::randperm< std::vector<int> >(n);

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
                            if (rom::numerical::invert(A, AInv)) {
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
                                            //ROM_COUT(Sm);
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

                        //ROM_COUT_VAR(qm);

                        BOOST_FOREACH(EdgePoint * e, childrens) {
                            //for( std::list<EdgePoint*>::iterator it = childrens.begin() ; it != childrens.end() ; )
                            //{
                            //ROM_COUT_VAR(numerical::distancePointEllipse( *p, qm, f )*1/std::sqrt((p->_normGrad)));
                            //ROM_COUT_VAR(numerical::distancePointEllipse( *p, qm, f )*1/(p->_normGrad));

                            double distFinal;

                            if (weightedType == NO_WEIGHT) {
                                distFinal = numerical::distancePointEllipse(*e, qm, f);
                            } else if (weightedType == INV_GRADIENT_WEIGHT) {
                                distFinal = numerical::distancePointEllipse(*e, qm, f)*255 / (e->_normGrad);
                            }
                            // todo@Lilian -- 1/grad, 1/sqrt(grad), 1 ??
                            //if( distFinal > threshold * Sm ) // 25.0 // 10.0
                            //{
                            //	it = childrens.erase(it);
                            //}else{
                            if (distFinal < threshold * Sm) {

                                filteredChildrens.push_back(e);
                                vDistFinal.push_back(distFinal);
                                //++it;
                            }

                            // l2.insert(l2.end(),l1.begin(),l1.end());

                        }
                        SmFinal = numerical::medianRef(vDistFinal);
                        //ROM_COUT(qm);
                    }
                }

                bool isAnotherSegment(numerical::geometry::Ellipse & outerEllipse, std::vector<EdgePoint*>& outerEllipsePoints, const std::vector<EdgePoint*>& filteredChildrens, const Candidate & anotherCandidate, std::vector< std::vector< Point2dN<double> > >& cctagPoints, std::size_t numCircles, double thrMedianDistanceEllipse) {
                    //ROM_COUT("Search for another segment");

                    using namespace boost::numeric::ublas;

                    const std::vector<EdgePoint*> & anotherOuterEllipsePoints = anotherCandidate._outerEllipsePoints;
                    //const std::list<EdgePoint*> & anotherFilteredChildrens = anotherCandidate._filteredChildrens;

                    numerical::geometry::Ellipse qm;
                    double Sm = 10000000000000.0;
                    const double f = 1.0;

                    // Create matrix containing homogeneous coordinates of filteredChildrens
                    //std::vector<double> weights;

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
                    //S1m = S2m = 1e13;

                    //ROM_COUT_VAR(S1m);
                    //ROM_COUT_VAR(S2m);

                    //std::cin.ignore().get();

                    while (cnt < 100)//50  ////////////
                    {
                        //ROM_COUT_VAR(S1m);
                        //ROM_COUT_VAR(S2m);

                        // Random subset of 5 points from pts
                        const std::vector<int> perm = rom::numerical::randperm< std::vector<int> >(outerEllipsePoints.size());

                        std::vector<rom::Point2dN< double > > points;


                        std::vector<int>::const_iterator it = perm.begin();
                        for (std::size_t i = 0; i < 4; ++i) {
                            points.push_back(Point2dN<double>(double(pts[*it](0)), double(pts[*it](1))));
                            ++it;
                        }

                        //	const std::size_t iEp = int(boost::math::round( std::max(double((std::rand()-1)), 0.0)/double(RAND_MAX)*double(anotherOuterEllipsePoints.size())-0.5+std::numeric_limits<double>::epsilon() ));

                        //points.push_back(Point2dN<double>(double(ePt->x()), double(ePt->y())));

                        //}else{
                        const std::vector<int> anotherPerm = rom::numerical::randperm< std::vector<int> >(anotherOuterEllipsePoints.size());

                        it = anotherPerm.begin();
                        for (std::size_t i = 0; i < 4; ++i) {
                            points.push_back(Point2dN<double>(double(anotherOuterEllipsePoints[*it]->x()), double(anotherOuterEllipsePoints[*it]->y())));
                            ++it;
                        }

                        numerical::geometry::Ellipse eToto;
                        rom::numerical::geometry::fitEllipse(points, eToto);

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
                            //ROM_COUT_VAR(dist.size());

                            //for(int iDist=0 ; iDist < dist.size() ; ++iDist)
                            //{
                            //	dist[iDist] = dist[iDist]*weights[iDist];
                            //}

                            const double S1 = numerical::medianRef(dist);
                            //ROM_COUT_VAR_DEBUG(q.matrix());
                            //ROM_COUT_VAR_DEBUG( S1 );

                            std::vector<double> anotherDist;
                            //ROM_COUT_VAR(anotherPts.size());
                            numerical::distancePointEllipse(anotherDist, anotherPts, q, f);
                            //ROM_COUT_VAR(anotherDist.size());

                            const double S2 = numerical::medianRef(anotherDist);
                            //ROM_COUT_VAR_DEBUG( S1 );

                            const double S = S1 + S2;

                            //ROM_COUT_VAR(S);
                            //ROM_COUT_VAR(S1);
                            //ROM_COUT_VAR(S2);

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

                    //ROM_COUT_DEBUG("[");
                    //BOOST_FOREACH( const EdgePoint* toto,  outerEllipsePoints){
                    //	//	outerEllipsePoints.push_back(toto);
                    //	ROM_COUT_DEBUG( "[" << toto->x() << "," << toto->y() << "] ; ");
                    //}
                    //ROM_COUT_DEBUG( "]" );

                    //ROM_COUT_DEBUG("[");
                    //BOOST_FOREACH( const EdgePoint* toto,  anotherOuterEllipsePoints){
                    //  //	outerEllipsePoints.push_back(toto);
                    //	ROM_COUT_DEBUG( "[" << toto->x() << "," << toto->y() << "] ; ");
                    //}
                    //ROM_COUT_DEBUG( "]" );

                    if (Sm < thr * SRef) {

                        ROM_COUT_DEBUG("Median test succed !!\n");

                        ROM_COUT_VAR_DEBUG(outerEllipse.matrix());
                        ROM_COUT_VAR_DEBUG(outerEllipsePoints.size());

                        std::vector<EdgePoint*> outerEllipsePointsTemp = outerEllipsePoints;
                        numerical::geometry::Ellipse outerEllipseTemp;

                        outerEllipsePointsTemp.insert(outerEllipsePointsTemp.end(), anotherOuterEllipsePoints.begin(), anotherOuterEllipsePoints.end());
                        ROM_COUT_VAR_DEBUG(outerEllipsePointsTemp.size());
                        // Compute the new ellipse which fits oulierEllipsePoints

                        //ROM_COUT("Before ellipse fitting");
                        //ROM_COUT_VAR(outerEllipse);
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

                                    ROM_COUT_VAR_DEBUG(outerEllipse);
                                    return true;
                                } else {
                                    ROM_COUT_DEBUG("isAnotherSegment : intermediate points outside or bad gradient orientations: ");
                                    return false;
                                }
                            } else {
                                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(FINAL_MEDIAN_TEST_FAILED_WHILE_ASSEMBLING);
                                ROM_COUT_DEBUG("SmFinal > thrMedianDistanceEllipse in isAnotherSegment");
                            }
                        } else {
                            CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(QUALITY_TEST_FAILED_WHILE_ASSEMBLING);
                            ROM_COUT_DEBUG("Quality too high: " << quality);
                            return false;
                        }
                    } else {
                        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(MEDIAN_TEST_FAILED_WHILE_ASSEMBLING);
                        ROM_COUT_DEBUG("Test failed !!\n");
                        return false;
                    }
                    return false;
                }

                /*
                void testOutlierRemoval()
                {
                        // I. Test de la fonction distance(vecteur, Ellipse)
                        bounded_vector<double, 3> vec(3);
                        vec(0) = 424;
                        vec(1) = 144;
                        vec(2) = 1;
                        bounded_vector<double, 5> param;
                        param(0) = 0;
                        param(1) = 0;
                        param(2) = 100.f;
                        param(3) = 200.f;
                        param(4) = boost::math::constants::pi<float>()/3;

                        Ellipse q = Ellipse(rom::Point2dN<double>(0.f,0.f), 100.f, 200.f, boost::math::constants::pi<float>()/3);
                        double f = 1.f;

                        double dist = distancePointEllipse(vec,  q, f);
                        std::cout << "dist : " << dist << std::endl ;

                        // II. Test de outlierRemoval
                        std::vector<EdgePoint*> childrens;
                        Label l;
                        // Données réelles
                        childrens.push_back(&EdgePoint(424, 142, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 144, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 145, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 146, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 147, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 155, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 156, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 157, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 158, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 159, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 160, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 161, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 162, 1, 1, &l));
                        childrens.push_back(&EdgePoint(424, 163, 1, 1, &l));
                        childrens.push_back(&EdgePoint(425, 165, 1, 1, &l));
                        childrens.push_back(&EdgePoint(425, 166, 1, 1, &l));
                        childrens.push_back(&EdgePoint(425, 167, 1, 1, &l));
                        childrens.push_back(&EdgePoint(430, 181, 1, 1, &l));
                        childrens.push_back(&EdgePoint(431, 123, 1, 1, &l));
                        childrens.push_back(&EdgePoint(431, 124, 1, 1, &l));
                        childrens.push_back(&EdgePoint(431, 182, 1, 1, &l));
                        childrens.push_back(&EdgePoint(431, 183, 1, 1, &l));
                        childrens.push_back(&EdgePoint(432, 122, 1, 1, &l));
                        childrens.push_back(&EdgePoint(432, 184, 1, 1, &l));
                        childrens.push_back(&EdgePoint(432, 185, 1, 1, &l));
                        childrens.push_back(&EdgePoint(433, 121, 1, 1, &l));
                        childrens.push_back(&EdgePoint(435, 188, 1, 1, &l));
                        childrens.push_back(&EdgePoint(435, 189, 1, 1, &l));
                        childrens.push_back(&EdgePoint(436, 118, 1, 1, &l));
                        childrens.push_back(&EdgePoint(436, 190, 1, 1, &l));
                        childrens.push_back(&EdgePoint(437, 117, 1, 1, &l));
                        childrens.push_back(&EdgePoint(437, 191, 1, 1, &l));
                        childrens.push_back(&EdgePoint(438, 117, 1, 1, &l));
                        childrens.push_back(&EdgePoint(438, 192, 1, 1, &l));
                        childrens.push_back(&EdgePoint(439, 116, 1, 1, &l));
                        childrens.push_back(&EdgePoint(439, 193, 1, 1, &l));
                        childrens.push_back(&EdgePoint(440, 116, 1, 1, &l));
                        childrens.push_back(&EdgePoint(440, 193, 1, 1, &l));
                        childrens.push_back(&EdgePoint(441, 115, 1, 1, &l));
                        childrens.push_back(&EdgePoint(230, 123, 1, 1, &l));
                        childrens.push_back(&EdgePoint(678, 23, 1, 1, &l));

                        // Display points before
                        for (int i=0; i<childrens.size() ; ++i)
                        {
                        std::cout << childrens[i]->vec << std::endl;
                        }

                        outlierRemoval(childrens);

                        // Display points after
                        std::cout << " Aprés outlierRemoval" << std::endl ;
                        for (int i=0; i<childrens.size() ; ++i)
                        {
                        std::cout << childrens[i]->vec << std::endl;
                        }
                }
                 */


            }
        }
    }
}


