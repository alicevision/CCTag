#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include "modeConfig.hpp"
#include "fileDebug.hpp"

#include "ellipse.hpp"
#include "detection.hpp"
#include "vote.hpp"
#include "visualDebug.hpp"
#include "multiresolution.hpp"
#include "miscellaneous.hpp"

#include "ellipseFittingWithGradient.hpp"
#include "CCTagFlowComponent.hpp"


#include <cctag/geometry/point.hpp>
#include <cctag/frame.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/identification.hpp>
#include <cctag/toolbox.hpp>
#include <cctag/toolbox/gilTools.hpp>
#include <cctag/types.hpp>
#include <cctag/image.hpp>
#include <cctag/canny.hpp>
#include <cctag/global.hpp>
#include <cctag/fileDebug.hpp>

#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/unordered/unordered_set.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <exception>
#include <fstream>
#include <list>
#include <utility>

namespace rom {
    namespace vision {
        namespace marker {
            namespace cctag {

void cctagDetectionFromEdges(
                        CCTag::List& markers,
                        std::vector<EdgePoint>& points,
                        const boost::gil::gray8_view_t & sourceView,
                        const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX,
                        const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
                        const EdgePointsImage& edgesMap,
                        const FrameId frame,
                        int pyramidLevel,
                        double scale,
                        const cctag::Parameters & params)
{
    POP_ENTER;
    using namespace boost::gil;
    boost::timer t;
    // Get vote winners
    WinnerMap winners;
    std::vector<EdgePoint*> seeds;


    // Voting procedure applied on every edge points. 
    vote(points, seeds, edgesMap, winners, params._distSearch, params._angleVoting, params._ratioVoting, params._numCrowns, params._minVotesToSelectCandidate, cannyGradX, cannyGradY);

    ///////////////////////////////////////// WRITING VOTE /////////////////////////
#if defined(DEBUG) || defined(CCTAG_STAT_DEBUG)
    {
        POP_INFO << "running optional 'voting' block" << std::endl;
        std::size_t mx = 0;
        boost::gil::gray8_image_t vimg(sourceView.dimensions());
        boost::gil::gray8_view_t votevw(view(vimg));

        // Zero filling
        boost::gil::fill_black(votevw);

        for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr) {
            EdgePoint* winner = itr->first;
            std::list<EdgePoint*> v = itr->second;
            if (mx < v.size()) {
                mx = v.size();
            }
        }

        for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr) {
                            EdgePoint* winner = itr->first;
                            std::list<EdgePoint*> v = itr->second;
                            *votevw.xy_at(winner->x(), winner->y()) = (unsigned char) ((v.size() * 10.0));
        }

        std::stringstream outFilenameVote;
        outFilenameVote << "voteLevel" << CCTagVisualDebug::instance().getPyramidLevel();
        CCTagVisualDebug::instance().initBackgroundImage(color_converted_view<rgb8_pixel_t>(votevw));
        CCTagVisualDebug::instance().newSession(outFilenameVote.str());

    }
#endif

    boost::timer t3;
    boost::posix_time::ptime tstart0(boost::posix_time::microsec_clock::local_time());

    std::size_t nSegmentOut = 0;

#ifdef CCTAG_STAT_DEBUG
    std::stringstream outFlowComponents;
    outFlowComponents << "flowComponentsLevel" << pyramidLevel << ".txt";
    CCTagFileDebug::instance().newSession(outFlowComponents.str());
#endif

    // sort candidates.
#ifdef GRIFF_DEBUG
    if( seeds.size() > 0 )
    {
        POP_INFO << "'optional' voting block is not really optional? " << seeds.size() << " seeds" << std::endl;
    }
    else
    {
        POP_INFO << "sorting empty edge point vector?" << std::endl;
    }
#endif // GRIFF_DEBUG
    std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);

    const std::size_t nSeedsToProcess = std::min(seeds.size(), params._maximumNbSeeds);

    std::list<Candidate> vCandidateLoopOne;

    //BOOST_FOREACH(EdgePoint* pmax, candidates)
    for (int iSeed = 0; iSeed < nSeedsToProcess; ++iSeed) {

        EdgePoint* seed = seeds[iSeed];

        if (!seed->_processedIn) {

            Candidate candidate;

            candidate._seed = seed;
            std::list<EdgePoint*> & convexEdgeSegment = candidate._convexEdgeSegment;
            edgeLinking(edgesMap, convexEdgeSegment, seed, winners, params._windowSizeOnInnerEllipticSegment, params._averageVoteMin);

            // Compute the average number of received points.
            int nReceivedVote = 0;
            int nVotedPoints = 0;

            BOOST_FOREACH(EdgePoint * p, convexEdgeSegment) {
                nReceivedVote += winners[p].size();
                if (winners[p].size() > 0){
                    ++nVotedPoints;
                }
            }
            candidate._averageReceivedVote = (float) (nReceivedVote*nReceivedVote) / (float) nVotedPoints;

            if (vCandidateLoopOne.size() > 0) {
                std::list<Candidate>::iterator it = vCandidateLoopOne.begin();
                while ( ((*it)._averageReceivedVote > candidate._averageReceivedVote) && ( it != vCandidateLoopOne.end() ) ) {
                    ++it;
                }
                vCandidateLoopOne.insert(it, candidate);
            } else {
                vCandidateLoopOne.push_back(candidate);
            }
        }
    }

    const std::size_t nCandidatesLoopOneToProcess = std::min(vCandidateLoopOne.size(), params._maximumNbCandidatesLoopTwo);

    std::vector<Candidate> vCandidateLoopTwo;
    vCandidateLoopTwo.reserve(nCandidatesLoopOneToProcess);

                    //BOOST_FOREACH(EdgePoint* pmax, candidates)

    std::list<Candidate>::iterator it = vCandidateLoopOne.begin();
    std::size_t iCandidate = 0;
                    
    while (iCandidate < nCandidatesLoopOneToProcess) {
                        
                        //for (int iCandidate = 0; iCandidate < nCandidatesLoopOneToProcess; ++iCandidate)
                        
        Candidate & candidate = *it;

        try {
            try {


                int xG = -1;
                int yG = -1;

                                //                                std::list<EdgePoint*> convexEdgeSegment;
                                //
                                //                                {
                                //                                    boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
                                //
                                //                                    edgeLinking(edgesMap, convexEdgeSegment, seed, winners, params._windowSizeOnInnerEllipticSegment, params._averageVoteMin);
                                //
                                //                                    boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                                //                                    boost::posix_time::time_duration d = tstop - tstart;
                                //                                    const double spendTime = d.total_milliseconds();
                                //                                    //ROM_TCOUT( "edgeLinking timer: " << spendTime );
                                //                                }


                std::list<EdgePoint*> childrens;

                {
                    boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
                    childrensOf(candidate._convexEdgeSegment, winners, childrens);
                    boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                    boost::posix_time::time_duration d = tstop - tstart;
                    const double spendTime = d.total_milliseconds();
                                    //ROM_TCOUT( "childrenOf timer: " << spendTime );
                }

                if (childrens.size() < params._minPointsSegmentCandidate) {
                    ROM_COUT_DEBUG(" childrens.size() < minPointsSegmentCandidate ");
                    ++it;
                    ++iCandidate;
                    continue;
                }

                candidate._score = childrens.size();

                double SmFinal = 1e+10;

                std::vector<EdgePoint*> & filteredChildrens = candidate._filteredChildrens;

                {
                    ROM_COUT_DEBUG("Before oulierRemoval!");

                    boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
                    outlierRemoval(childrens, filteredChildrens, SmFinal, params._threshRobustEstimationOfOuterEllipse, cctag::kWeight);
                    boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                    boost::posix_time::time_duration d = tstop - tstart;
                    const double spendTime = d.total_milliseconds();
                }

                if ((candidate._seed->x() == xG) && (candidate._seed->y() == yG)) {
                    coutVPoint(filteredChildrens);
                    ROM_PAUSE
                }

                                // New criterion -- to be tunned todo@Lilian -- tres important pour l'optimisation - learning


                if (filteredChildrens.size() < 5) //todo@lilian see the case in outlierRemoval where filteredChildrens.size()==0
                {
                    ROM_COUT_DEBUG(" filteredChildrens.size() < 5 ");
                    ++it;
                    ++iCandidate;
                    continue;
                }

                std::size_t nLabel = -1;

                {
                    std::size_t nSegmentCommon = -1;

                    BOOST_FOREACH(EdgePoint * p, filteredChildrens) {
                        if (p->_nSegmentOut != -1) {
                            nSegmentCommon = p->_nSegmentOut;
                            break;
                        }
                    }

                    if (nSegmentCommon == -1) {
                        nLabel = nSegmentOut;
                        ++nSegmentOut;
                    } else {
                        nLabel = nSegmentCommon;
                    }

                    BOOST_FOREACH(EdgePoint * p, filteredChildrens) {
                        p->_nSegmentOut = nLabel;
                    }
                }


                std::vector<EdgePoint*> & outerEllipsePoints = candidate._outerEllipsePoints;
                rom::numerical::geometry::Ellipse & outerEllipse = candidate._outerEllipse;

                bool goodInit = false;

                {
                    boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
                    goodInit = ellipseGrowingInit(points, filteredChildrens, outerEllipse);

                    boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                    boost::posix_time::time_duration d = tstop - tstart;
                    const double spendTime = d.total_milliseconds();
                }

                if ((candidate._seed->x() == xG) && (candidate._seed->y() == yG)) {
                    ROM_COUT("after egInit");
                    ROM_PAUSE
                }

                t.restart();

                {
                    boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
                                    //ellipseGrowing(edgesMap, filteredChildrens, outerEllipsePoints, outerEllipse, params._ellipseGrowingEllipticHullWidth, nSegmentOut, nLabel, goodInit);
                    ellipseGrowing2(edgesMap, filteredChildrens, outerEllipsePoints, outerEllipse, params._ellipseGrowingEllipticHullWidth, nSegmentOut, nLabel, goodInit);

                    candidate._nLabel = nLabel;

                    boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                    boost::posix_time::time_duration d = tstop - tstart;
                    const double spendTime = d.total_milliseconds();
                }

                if ((candidate._seed->x() == xG) && (candidate._seed->y() == yG)) {
                    coutVPoint(outerEllipsePoints);
                    ROM_COUT_DEBUG("after eg");
                    ROM_PAUSE
                }

                                // {

                                    //boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
                                    //outlierRemoval( outerEllipsePoints, SmFinal, threshRobustEstimationOfOuterEllipse);
                                    //boost::posix_time::ptime tstop( boost::posix_time::microsec_clock::local_time() );
                                    //boost::posix_time::time_duration d = tstop - tstart;
                                    //const double spendTime = d.total_milliseconds();
                                // }


                                // New test -- to be tunned todo@Lilian
                std::vector<double> vDistFinal;
                vDistFinal.clear();
                vDistFinal.reserve(outerEllipsePoints.size());

                                // Rajouter une inline function pour les egdePoints*, des pbs d'heritage, c'est le bordel pour les fonctions distancePointEllipse.

                double distMax = 0;

                BOOST_FOREACH(EdgePoint * p, outerEllipsePoints) {
                    double distFinal = numerical::distancePointEllipse(*p, outerEllipse, 1.0);
                    vDistFinal.push_back(distFinal);

                    if (distFinal > distMax) {
                        distMax = distFinal;
                    }

                }

                                // todo@Lilian : sort => need to be replace by nInf
                SmFinal = numerical::medianRef(vDistFinal);

                                //const double thrMedianDistanceEllipse = 3;

                                //ROM_COUT_VAR( SmFinal );
                if (SmFinal > params._thrMedianDistanceEllipse) {
                    ROM_COUT_DEBUG("SmFinal < params._thrMedianDistanceEllipse -- after ellipseGrowing");
                    ++it;
                    ++iCandidate;
                    continue;
                }

                double quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);
                if (quality > 1.1)//0.4 5.0 0.2
                {
                    ROM_COUT_DEBUG("Quality too high!");
                    ++it;
                    ++iCandidate;
                    continue;
                }


                double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();
                if ((ratioSemiAxes < 0.05) || (ratioSemiAxes > 20)) //0.4 5.0 0.2
                {
                    ROM_COUT_DEBUG("Too high ratio between semi-axes!");
                    ++it;
                    ++iCandidate;
                    continue;
                }

                if ((candidate._seed->x() == xG) && (candidate._seed->y() == yG)) {
                    ROM_COUT("end");
                    ROM_PAUSE
                }

                vCandidateLoopTwo.push_back(candidate);

#ifdef CCTAG_STAT_DEBUG
                // Add childrens to output the filtering results (from outlierRemoval)
                vCandidateLoopTwo.back().setChildrens(childrens);

                // Write all selectedFlowComponent
                CCTagFlowComponent flowComponent(outerEllipsePoints, childrens, filteredChildrens, outerEllipse, candidate._convexEdgeSegment, *(candidate._seed), params._nCircles);
                CCTagFileDebug::instance().outputFlowComponentInfos(flowComponent);
#endif

            } catch (cv::Exception& e) {
                ROM_COUT(" Opencv : exception levee !!  ");
                const char* err_msg = e.what();
                                //std::cout << "exception caught: " << err_msg << std::endl;
                                // Ellipse fitting don't pass.
            }
        } catch (...) {
            ROM_COUT(" Exception levee, loop 1 !");
        }

        ++it;
        ++iCandidate;

    }

    /// Debug
    ROM_COUT_DEBUG("__________________________ Liste des germes ___________________________________");

    ROM_COUT_VAR(vCandidateLoopTwo.size());

    BOOST_FOREACH(const Candidate & anotherCandidate, vCandidateLoopTwo) {
        ROM_COUT_DEBUG("X = [ " << anotherCandidate._seed->x() << " , " << anotherCandidate._seed->y() << "]");
    }


    boost::posix_time::ptime tstop1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration d1 = tstop1 - tstart0;
    const double spendTime1 = d1.total_milliseconds();
    ROM_COUT(" ============ TIME FOR THE 1ST LOOP ============= " << spendTime1 << " ms");

#if defined CCTAG_STAT_DEBUG && defined DEBUG
    std::stringstream outFlowComponentsAssembling;
    outFlowComponentsAssembling << "flowComponentsAssemblingLevel" << pyramidLevel << ".txt";
    CCTagFileDebug::instance().newSession(outFlowComponentsAssembling.str());
    CCTagFileDebug::instance().initFlowComponentsIndex(2);
#endif

    BOOST_FOREACH(const Candidate & candidate, vCandidateLoopTwo) {
#ifdef CCTAG_STAT_DEBUG
        std::vector<Candidate> componentCandidates;
#ifdef DEBUG
        CCTagFileDebug::instance().resetFlowComponent();
#endif
#endif

        // recopie Hyper couteux, trouver une autre solution !! todo@Lilian

        std::vector<EdgePoint*> outerEllipsePoints = candidate._outerEllipsePoints;
        rom::numerical::geometry::Ellipse outerEllipse = candidate._outerEllipse;
        std::vector<EdgePoint*> filteredChildrens = candidate._filteredChildrens;

        std::vector< std::vector< Point2dN<double> > > cctagPoints;

        try {

            // SEARCH FOR ANOTHER SEGMENT
            double quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);

            // if the quality is greater than 1/3
            if (params._searchForAnotherSegment) {

                boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());

                if ((quality > 0.25) && (quality < 0.7)) { //extraInfos

                    ROM_COUT_DEBUG("================= Look for another segment ==================");

                    int score = -1;
                    int iMax = 0;
                    int i = 0;

                    double ratioExpension = 2.5;
                    numerical::geometry::Cercle circularResearchArea(Point2dN<double>(candidate._seed->x(), candidate._seed->y()), candidate._seed->_flowLength * ratioExpension);

                    // Search for another segment

                    BOOST_FOREACH(const Candidate & anotherCandidate, vCandidateLoopTwo) {
                        if (&candidate != &anotherCandidate) {
                            if (candidate._nLabel != anotherCandidate._nLabel) {
                                if ((anotherCandidate._seed->_flowLength / candidate._seed->_flowLength > 0.666) && (anotherCandidate._seed->_flowLength / candidate._seed->_flowLength < 1.5)) {
                                    if (isInEllipse(circularResearchArea, rom::Point2dN<double>(double(anotherCandidate._seed->x()), double(anotherCandidate._seed->y())))) {
                                        if (anotherCandidate._score > score) {
                                            score = anotherCandidate._score;
                                            iMax = i;
                                        }
                                    } else {
                                        CCTagFileDebug::instance().setResearchArea(circularResearchArea);
                                        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(NOT_IN_RESEARCH_AREA);
                                    }
                                } else {
                                    CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(FLOW_LENGTH);
                                }
                            } else {
                                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(SAME_LABEL);
                            }
                        }
                        ++i;
#if defined CCTAG_STAT_DEBUG && defined DEBUG
                        if (i < vCandidateLoopTwo.size()) {
                            CCTagFileDebug::instance().incrementFlowComponentIndex(1);
                        }
#endif
                    }

                    ROM_COUT_VAR_DEBUG(iMax);
                    ROM_COUT_VAR_DEBUG(*(vCandidateLoopTwo[iMax])._seed);

                    if (score > 0) {

                        const Candidate & selectedCandidate = vCandidateLoopTwo[iMax];

                        ROM_COUT_VAR_DEBUG(selectedCandidate._outerEllipse);

                        if (isAnotherSegment(outerEllipse, outerEllipsePoints, selectedCandidate._filteredChildrens, selectedCandidate, cctagPoints, params._numCrowns * 2, params._thrMedianDistanceEllipse)) {

                            quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);

#ifdef CCTAG_STAT_DEBUG
                            componentCandidates.push_back(selectedCandidate);
                            // Assembling succeed !
#endif                                      
                            CCTagFileDebug::instance().setFlowComponentAssemblingState(true, iMax);
                        }
                    }
                }

                boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
                boost::posix_time::time_duration d = tstop - tstart;
                const double spendTime = d.total_milliseconds();
            }

            // Init cctag points  // Add intermediary points - required ? todo@Lilian
            if (!addCandidateFlowtoCCTag(candidate._filteredChildrens, candidate._outerEllipsePoints, outerEllipse, cctagPoints, params._numCrowns * 2)) {
                ROM_COUT_DEBUG("Points outside the outer ellipse OR CCTag not valid : bad gradient orientations");
                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTSOUTSIDE_OR_BADGRADORIENT);
                CCTagFileDebug::instance().incrementFlowComponentIndex(0);
                continue;
            } else {
#ifdef CCTAG_STAT_DEBUG
                componentCandidates.push_back(candidate);
#endif                          
                ROM_COUT_DEBUG("Points inside the outer ellipse and good gradient orientations");
            }
            // Create ellipse with its real size from original image.
            rom::numerical::geometry::Ellipse rescaleEllipse(outerEllipse.center(), outerEllipse.a() * scale, outerEllipse.b() * scale, outerEllipse.angle());
            int realPixelPerimeter = rasterizeEllipsePerimeter(rescaleEllipse);

            double realSizeOuterEllipsePoints = quality*realPixelPerimeter;

                            // Naive reject condition todo@Lilian
                            /*if ( ( ( quality <= 0.35 ) && ( realSizeOuterEllipsePoints >= 300.0 ) ) ||//0.35
                                     ( ( quality <= 0.45 ) && ( realSizeOuterEllipsePoints >= 200.0 ) && ( realSizeOuterEllipsePoints < 300.0 ) ) ||//0.45
                                     ( ( quality <= 0.50 ) && ( realSizeOuterEllipsePoints >= 100.0 ) && ( realSizeOuterEllipsePoints < 200.0 ) ) ||//0.50
                                     ( ( quality <= 0.50 ) && ( realSizeOuterEllipsePoints >= 70.0  ) && ( realSizeOuterEllipsePoints < 100.0 ) ) ||//0.5
                                     ( ( quality <= 0.96 ) && ( realSizeOuterEllipsePoints >= 50.0  ) && ( realSizeOuterEllipsePoints < 70.0 ) ) ||//0.96
                                     ( realSizeOuterEllipsePoints < 50.0  ) )
                            {
                                    ROM_COUT_DEBUG( "Not enough outer ellipse points : realSizeOuterEllipsePoints : " << realSizeOuterEllipsePoints << ", rasterizeEllipsePerimeter : " << rasterizeEllipsePerimeter( outerEllipse )*scale << ", quality : " << quality );
                                    continue;
                            }*/

                            // TOTO 2nd outlier removal on the outer ellipse points
                            /* {
                               std::vector<EdgePoint*> childrens2;
                               outlierRemoval(childrens, childrens2);
                               } */

            rom::Point2dN<double> markerCenter;
            rom::numerical::BoundedMatrix3x3d markerHomography;

            const double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();

            if (ratioSemiAxes > 8.0 || ratioSemiAxes < 0.125) {
                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RATIO_SEMIAXIS);
                CCTagFileDebug::instance().incrementFlowComponentIndex(0);
                ROM_COUT_DEBUG("Too high ratio between semi-axes!");
                continue;
            }

            std::vector<double> vDistFinal;
            vDistFinal.clear();
            vDistFinal.reserve(outerEllipsePoints.size());

                            // todo@Lilian
                            // Rajouter une inline function pour les egdePoints*,
                            // des pbs d'heritage, c'est le bordel pour les fonctions
                            // distancePointEllipse.

            double resSquare = 0;
            double distMax = 0;

            BOOST_FOREACH(EdgePoint * p, outerEllipsePoints) {
                double distFinal = numerical::distancePointEllipse(*p, outerEllipse, 1.0);
                resSquare += distFinal; //*distFinal;

                if (distFinal > distMax) {
                    distMax = distFinal;
                }
            }

            resSquare = sqrt(resSquare);
            resSquare /= outerEllipsePoints.size();


            numerical::geometry::Ellipse qIn, qOut;
            computeHull(outerEllipse, 3.6, qIn, qOut);

            bool isValid = true;

            BOOST_FOREACH(const EdgePoint * p, outerEllipsePoints) {
                if (!isInHull(qIn, qOut, p)) {
                    isValid = false;
                    break;
                }
            }
            if (!isValid) {
                CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTS_OUTSIDE_ELLHULL);
                CCTagFileDebug::instance().incrementFlowComponentIndex(0);

                ROM_COUT_DEBUG("Distance max to high!");
                continue;
            }

            std::vector< Point2dN<int> > vPoint;

            double quality2 = 0;

            BOOST_FOREACH(const EdgePoint* p, outerEllipsePoints) {
                quality2 += p->_normGrad;
            }

            quality2 *= scale;

            markers.push_back(new CCTag(-1, outerEllipse.center(), cctagPoints, outerEllipse, markerHomography, pyramidLevel, scale, quality2));
#ifdef CCTAG_STAT_DEBUG
            markers.back().setFlowComponents(componentCandidates);
#ifdef DEBUG 
            CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PASS_ALLTESTS);
            CCTagFileDebug::instance().incrementFlowComponentIndex(0);
#endif                          
#endif

            bool displayPushedMarker = false;

            if (displayPushedMarker) {
                ROM_COUT_VAR_DEBUG(outerEllipse);
                ROM_COUT_VAR_DEBUG(quality);
                coutVPoint(outerEllipsePoints);
                                //std::cin.ignore().get();
            }
                            //
            ROM_COUT_DEBUG("------------------------------Added marker------------------------------");
        } catch (...) {
            CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RAISED_EXCEPTION);
            CCTagFileDebug::instance().incrementFlowComponentIndex(0);
            // Ellipse fitting don't pass.
            ROM_COUT_CURRENT_EXCEPTION;
            ROM_COUT_DEBUG("Exception levee !");
        }
    }

    boost::posix_time::ptime tstop2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration d2 = tstop2 - tstop1;
    const double spendTime2 = d2.total_milliseconds();
    ROM_COUT(" ============ TIME FOR THE 1ST LOOP ============= " << spendTime2 << " ms");

                    //	markers.sort();

    ROM_COUT_DEBUG("Markers creation time: " << t3.elapsed());
    POP_LEAVE;
}

            } // namespace cctag

void cctagDetection(CCTag::List& markers, const FrameId frame, const boost::gil::rgb8_view_t& srcView, const cctag::Parameters & params, const bool bDisplayEllipses)
{
    POP_ENTER;
                using namespace cctag;
                using namespace boost::numeric::ublas;
                using namespace boost::gil;
                //	using namespace rom::img;

                std::srand(1);

                static const CCTagMarkersBank bank(params._cctagBankFilename);

                boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());

                // Grayscale transform
                gray8_image_t grayImg;
                gray8_view_t graySrc = rom::img::toGray(srcView, grayImg);

                // Compute canny
                typedef kth_channel_view_type<0, rgb32f_view_t>::type CannyView;
                typedef kth_channel_view_type<1, rgb32f_view_t>::type GradXView;
                typedef kth_channel_view_type<2, rgb32f_view_t>::type GradYView;

                rgb32f_image_t cannyRGBImg(graySrc.width(), graySrc.height());
                rgb32f_view_t cannyRGB(view(cannyRGBImg));
                CannyView cannyView;
                GradXView cannyGradX;
                GradYView cannyGradY;

                cannyView = kth_channel_view<0>(cannyRGB);
                // x gradient
                cannyGradX = kth_channel_view<1>(cannyRGB);
                // y gradient
                cannyGradY = kth_channel_view<2>(cannyRGB);

                cctag::cannyCv(graySrc, cannyRGB, cannyView, cannyGradX, cannyGradY, params._cannyThrLow, params._cannyThrHigh);

                cctagMultiresDetection(markers, graySrc, cannyRGB, frame, params);

                boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());

                boost::posix_time::time_duration d = tstop - tstart;
                const double spendTime = d.total_milliseconds();
                ROM_COUT("Etape de detection" << spendTime);

                ////////// Identification pass
                // To decomment -- enable cuts selection, homography computation and identification
                if (1) {
                    ROM_COUT_DEBUG("Before identify!");
                    CCTag::List::iterator it = markers.begin();
                    while (it != markers.end()) {
                        CCTag & cctag = *it;

                        //tstart( boost::posix_time::microsec_clock::local_time() );
                        const int detected = rom::vision::marker::identify(cctag, bank.getMarkers(), graySrc, cannyGradX, cannyGradY, params._numCrowns, params._numCutsInIdentStep, params._numSamplesOuterEdgePointsRefinement, params._cutsSelectionTrials, params._sampleCutLength, params._minIdentProba, params._useLMDif);

                        cctag.setStatus(detected);

                        //if ( detected < 0 )
                        //{
                        //	// Erase marker we were unable to indentify
                        //	it = markers.erase( it );
                        //}
                        //else
                        //{
                        try {
                            std::vector<rom::numerical::geometry::Ellipse> & ellipses = cctag.ellipses();

                            bounded_matrix<double, 3, 3> mInvH;
                            rom::numerical::invert(cctag.homography(), mInvH);

                            BOOST_FOREACH(double radiusRatio, cctag.radiusRatios()) {
                                //ROM_COUT_VAR_DEBUG(radiusRatio);
                                rom::numerical::geometry::Cercle circle(1.0 / radiusRatio);
                                ellipses.push_back(rom::numerical::geometry::Ellipse(prec_prod(trans(mInvH), prec_prod<bounded_matrix<double, 3, 3> >(circle.matrix(), mInvH))));
                            }

                            // Push the outer ellipse
                            ellipses.push_back(cctag.rescaledOuterEllipse());

                            //ROM_COUT_VAR_DEBUG( cctag.ellipses().size() );

                            ROM_COUT_VAR_DEBUG(cctag.id());
                            ++it;
                        } catch (...) {
                            // Impossible to construct Ellipse from computed homographies => conics are not ellipses!
                            //it = markers.erase( it );
                        }
                        //}
                    }
                }

                markers.sort();

                CCTagVisualDebug::instance().writeIdentificationView(markers);

                CCTagFileDebug::instance().newSession("identification.txt");

                BOOST_FOREACH(const CCTag & marker, markers) {
                    CCTagFileDebug::instance().outputMarkerInfos(marker);
                }

                //CCTagVisualDebug::instance().initBackgroundImage(srcView);
                //CCTagVisualDebug::instance().newSession("Identification");
                //
                //BOOST_FOREACH(const CCTag & marker, markers) {
                //    CCTagVisualDebug::instance().drawMarker(marker, false);
                //    CCTagVisualDebug::instance().drawInfos(marker, false);
                //}

                //CCTagVisualDebug::instance().outPutAllSessions();

    POP_LEAVE;
}
        } // namespace marker
    } // namespace vision
} // namespace rom
