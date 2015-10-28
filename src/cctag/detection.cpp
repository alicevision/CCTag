#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>

#include <cctag/fileDebug.hpp>
#include <cctag/ellipseGrowing.hpp>
#include <cctag/detection.hpp>
#include <cctag/vote.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/multiresolution.hpp>
#include <cctag/miscellaneous.hpp>
#include <cctag/ellipseFittingWithGradient.hpp>
#include <cctag/CCTagFlowComponent.hpp>
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
#include <cctag/talk.hpp> // for DO_TALK macro
#include <cctag/fileDebug.hpp>
#include "cuda/tag.h"

#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/unordered/unordered_set.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <exception>
#include <fstream>
#include <sstream>
#include <list>
#include <utility>

namespace cctag
{

void constructFlowComponentFromSeed(
        EdgePoint * seed,
        const EdgePointsImage& edgesMap,
        WinnerMap & winners, 
        std::list<Candidate> & vCandidateLoopOne,
        const Parameters & params)
{
  // Check if the seed has already been processed, i.e. belongs to an already
  // reconstructed flow component.
  if (!seed->_processedIn)
  {

    Candidate candidate;

    candidate._seed = seed;
    std::list<EdgePoint*> & convexEdgeSegment = candidate._convexEdgeSegment;

    // Convex edge linking from the seed in both directions. The linking
    // is performed until the convexity is lost.
    edgeLinking(edgesMap, convexEdgeSegment, seed, winners, 
            params._windowSizeOnInnerEllipticSegment, params._averageVoteMin);

    // Compute the average number of received points.
    int nReceivedVote = 0;
    int nVotedPoints = 0;

    BOOST_FOREACH(EdgePoint * p, convexEdgeSegment)
    {
      nReceivedVote += winners[p].size();
      if (winners[p].size() > 0)
      {
        ++nVotedPoints;
      }
    }

    // All flow components WILL BE next sorted based on this characteristic (scalar)
    candidate._averageReceivedVote = (float) (nReceivedVote*nReceivedVote) / (float) nVotedPoints;

    if (vCandidateLoopOne.size() > 0)
    {
      std::list<Candidate>::iterator it = vCandidateLoopOne.begin();
      while ( ((*it)._averageReceivedVote > candidate._averageReceivedVote) 
              && ( it != vCandidateLoopOne.end() ) )
      {
        ++it;
      }
      vCandidateLoopOne.insert(it, candidate);
    }
    else
    {
      vCandidateLoopOne.push_back(candidate);
    }
  }
}

void completeFlowComponent(
        Candidate & candidate,
        WinnerMap & winners,
        std::vector<EdgePoint> & points,
        const EdgePointsImage& edgesMap,
        std::vector<Candidate> & vCandidateLoopTwo,
        std::size_t & nSegmentOut,
        const Parameters & params)
{
  try
  {
    try
    {
      std::list<EdgePoint*> childrens;

      childrensOf(candidate._convexEdgeSegment, winners, childrens);

      if (childrens.size() < params._minPointsSegmentCandidate)
      {
        DO_TALK( CCTAG_COUT_DEBUG(" childrens.size() < minPointsSegmentCandidate "); ) return;
      }

      candidate._score = childrens.size();

      double SmFinal = 1e+10;

      std::vector<EdgePoint*> & filteredChildrens = candidate._filteredChildrens;

      outlierRemoval(childrens, filteredChildrens, SmFinal, 
              params._threshRobustEstimationOfOuterEllipse, kWeight);

      // todo@lilian see the case in outlierRemoval
      // where filteredChildrens.size()==0
      if (filteredChildrens.size() < 5)
      {
        DO_TALK( CCTAG_COUT_DEBUG(" filteredChildrens.size() < 5 "); )
        return;
      }

      std::size_t nLabel = -1;

      {
        ssize_t nSegmentCommon = -1; // std::size_t nSegmentCommon = -1;

        BOOST_FOREACH(EdgePoint * p, filteredChildrens)
        {
          if (p->_nSegmentOut != -1)
          {
            nSegmentCommon = p->_nSegmentOut;
            break;
          }
        }

        if (nSegmentCommon == -1)
        {
          nLabel = nSegmentOut;
          ++nSegmentOut;
        }
        else
        {
          nLabel = nSegmentCommon;
        }

        BOOST_FOREACH(EdgePoint * p, filteredChildrens)
        {
          p->_nSegmentOut = nLabel;
        }
      }

      std::vector<EdgePoint*> & outerEllipsePoints = candidate._outerEllipsePoints;
      cctag::numerical::geometry::Ellipse & outerEllipse = candidate._outerEllipse;

      bool goodInit = false;

      goodInit = ellipseGrowingInit(points, filteredChildrens, outerEllipse);

      ellipseGrowing2(edgesMap, filteredChildrens, outerEllipsePoints, outerEllipse,
                      params._ellipseGrowingEllipticHullWidth, nSegmentOut, 
                      nLabel, goodInit);

      candidate._nLabel = nLabel;

      std::vector<double> vDistFinal;
      vDistFinal.clear();
      vDistFinal.reserve(outerEllipsePoints.size());

      // Clean point (egdePoints*) to ellipse distance with inheritance 
      // -- the current solution is dirty --
      double distMax = 0;

      BOOST_FOREACH(EdgePoint * p, outerEllipsePoints)
      {
        double distFinal = numerical::distancePointEllipse(*p, outerEllipse, 1.0);
        vDistFinal.push_back(distFinal);

        if (distFinal > distMax)
        {
          distMax = distFinal;
        }
      }

      // todo@Lilian : sort => need to be replace by nInf
      SmFinal = numerical::medianRef(vDistFinal);

      if (SmFinal > params._thrMedianDistanceEllipse)
      {
        DO_TALK( CCTAG_COUT_DEBUG("SmFinal < params._thrMedianDistanceEllipse -- after ellipseGrowing"); )
        return;
      }

      double quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);
      if (quality > 1.1)
      {
        DO_TALK( CCTAG_COUT_DEBUG("Quality too high!"); )
        return;
      }

      double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();
      if ((ratioSemiAxes < 0.05) || (ratioSemiAxes > 20))
      {
        DO_TALK( CCTAG_COUT_DEBUG("Too high ratio between semi-axes!"); )
        return;
      }

      vCandidateLoopTwo.push_back(candidate);

#ifdef CCTAG_SERIALIZE
      // Add childrens to output the filtering results (from outlierRemoval)
      vCandidateLoopTwo.back().setChildrens(childrens);

      // Write all selectedFlowComponent
      CCTagFlowComponent flowComponent(outerEllipsePoints, childrens, filteredChildrens,
                                       outerEllipse, candidate._convexEdgeSegment,
                                      *(candidate._seed), params._nCircles);
      CCTagFileDebug::instance().outputFlowComponentInfos(flowComponent);
#endif

    }
    catch (cv::Exception& e)
    {
      DO_TALK( CCTAG_COUT_DEBUG( "OpenCV exception" ); )
      e.what();
    }
  }
  catch (...)
  {
    DO_TALK( CCTAG_COUT_DEBUG( "Exception raised in the second main loop." ); )
  }
}

/* Brief: Aims to assemble two flow components if they lie on the same image CCTag
 quality: quality of the outer ellipse (fraction of the ellipse perimeter covered 
 by the extracted edge points, i.e. those contained in outerEllipsePoints
 candidate: flow component being assembled
 vCandidateLoopTwo: set of all flow component detected in the image
 outerEllipse: outerEllipse of the flow component being assembled
 outerEllipsePoints: edge points lying extracted on the outer ellipse
 cctagPoints: set of points constituting the final cctag 
 params: parameters of the system's algorithm */
void flowComponentAssembling(
        double & quality,
        const Candidate & candidate,
        const std::vector<Candidate> & vCandidateLoopTwo,
        numerical::geometry::Ellipse & outerEllipse,
        std::vector<EdgePoint*>& outerEllipsePoints,
        std::vector< std::vector< DirectedPoint2d<double> > >& cctagPoints,
        const Parameters & params
#ifndef CCTAG_SERIALIZE
        )
#else
        , std::vector<Candidate> & componentCandidates)
#endif
{
  boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
  
  DO_TALK( CCTAG_COUT_DEBUG("================= Look for another segment =================="); )

  int score = -1;
  int iMax = 0;
  int i = 0;

  double ratioExpension = 2.5;
  numerical::geometry::Cercle circularResearchArea(
         Point2dN<double>( candidate._seed->x(), candidate._seed->y() ),
         candidate._seed->_flowLength * ratioExpension);

  // Search for another segment
  BOOST_FOREACH(const Candidate & anotherCandidate, vCandidateLoopTwo)
  {
    if (&candidate != &anotherCandidate)
    {
      if (candidate._nLabel != anotherCandidate._nLabel)
      {
        if ((anotherCandidate._seed->_flowLength / candidate._seed->_flowLength > 0.666)
                && (anotherCandidate._seed->_flowLength / candidate._seed->_flowLength < 1.5))
        {
          if (isInEllipse(circularResearchArea, 
                  cctag::Point2dN<double>(double(anotherCandidate._seed->x()), double(anotherCandidate._seed->y()))))
          {
            if (anotherCandidate._score > score)
            {
              score = anotherCandidate._score;
              iMax = i;
            }
          }
          else
          {
            CCTagFileDebug::instance().setResearchArea(circularResearchArea);
            CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(NOT_IN_RESEARCH_AREA);
          }
        }
        else
        {
          CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(FLOW_LENGTH);
        }
      }
      else
      {
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(SAME_LABEL);
      }
    }
    ++i;
#if defined CCTAG_SERIALIZE && defined DEBUG
    if (i < vCandidateLoopTwo.size())
    {
      CCTagFileDebug::instance().incrementFlowComponentIndex(1);
    }
#endif
  }

  DO_TALK( CCTAG_COUT_VAR_DEBUG(iMax); )
  DO_TALK( CCTAG_COUT_VAR_DEBUG(*(vCandidateLoopTwo[iMax])._seed); )

  if (score > 0)
  {
    const Candidate & selectedCandidate = vCandidateLoopTwo[iMax];
    DO_TALK( CCTAG_COUT_VAR_DEBUG(selectedCandidate._outerEllipse); )

    if( isAnotherSegment(outerEllipse, outerEllipsePoints, 
            selectedCandidate._filteredChildrens, selectedCandidate,
            cctagPoints, params._nCrowns * 2,
            params._thrMedianDistanceEllipse) )
    {
      quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);

#ifdef CCTAG_SERIALIZE
      componentCandidates.push_back(selectedCandidate);
#endif
      CCTagFileDebug::instance().setFlowComponentAssemblingState(true, iMax);
    }
  }

  boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d = tstop - tstart;
  d.total_milliseconds();
}


void cctagDetectionFromEdges(
        CCTag::List&            markers,
        std::vector<EdgePoint>& points,
        const cv::Mat&          src,
        WinnerMap&              winners,
        const std::vector<EdgePoint*>& seeds,
        const EdgePointsImage& edgesMap,
        const std::size_t frame,
        int pyramidLevel,
        double scale,
        const Parameters & params)
{
  // using namespace boost::gil;

  // Call for debug only. Write the vote result as an image.
  createImageForVoteResultDebug(src, winners, pyramidLevel); //todo@Lilian: change this function to put a cv::Mat as input.

  // Set some timers
  boost::timer t3;
  boost::posix_time::ptime tstart0(boost::posix_time::microsec_clock::local_time());

  std::size_t nSegmentOut = 0;

#ifdef CCTAG_SERIALIZE
  std::stringstream outFlowComponents;
  outFlowComponents << "flowComponentsLevel" << pyramidLevel << ".txt";
  CCTagFileDebug::instance().newSession(outFlowComponents.str());
#endif

  if( seeds.size() <= 0 )
  {
    // No seeds to process
    return;
  }

  const std::size_t nMaximumNbSeeds = std::max(src.rows/2, (int) params._maximumNbSeeds);
  
  const std::size_t nSeedsToProcess = std::min(seeds.size(), nMaximumNbSeeds);

  std::list<Candidate> vCandidateLoopOne;

  // Process all the nSeedsToProcess-first seeds.
  // In the following loop, a seed will lead to a flow component if it lies
  // on the inner ellipse of a CCTag.
  // The edge points lying on the inner ellipse and their voters (lying on the outer ellipse
  // will be collected and constitute the initial data of a flow component.
  for (int iSeed = 0; iSeed < nSeedsToProcess; ++iSeed)
  {
    constructFlowComponentFromSeed(seeds[iSeed], edgesMap, winners, vCandidateLoopOne, params);
  }

  const std::size_t nFlowComponentToProcessLoopTwo = 
          std::min(vCandidateLoopOne.size(), params._maximumNbCandidatesLoopTwo);

  std::vector<Candidate> vCandidateLoopTwo;
  vCandidateLoopTwo.reserve(nFlowComponentToProcessLoopTwo);

  std::list<Candidate>::iterator it = vCandidateLoopOne.begin();
  std::size_t iCandidate = 0;

  // Second main loop:
  // From the flow components selected in the first loop, the outer ellipse will
  // be here entirely recovered.
  // The GPU implementation should stop at this point => layers ->  EdgePoint* creation.

  while (iCandidate < nFlowComponentToProcessLoopTwo)
  {
    completeFlowComponent(*it, winners, points, edgesMap, vCandidateLoopTwo, nSegmentOut, params);
    ++it;
    ++iCandidate;
  }

  DO_TALK(
    CCTAG_COUT_VAR_DEBUG(vCandidateLoopTwo.size());
    CCTAG_COUT_DEBUG("================= List of seeds =================");
    BOOST_FOREACH(const Candidate & anotherCandidate, vCandidateLoopTwo)
    {
      CCTAG_COUT_DEBUG("X = [ " << anotherCandidate._seed->x() << " , " << anotherCandidate._seed->y() << "]");
    }
  )

  boost::posix_time::ptime tstop1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d1 = tstop1 - tstart0;
  const double spendTime1 = d1.total_milliseconds();
  DO_TALK( CCTAG_COUT_OPTIM(" Time in the 1st loop " << spendTime1 << " ms"); )

#if defined CCTAG_SERIALIZE && defined DEBUG
  std::stringstream outFlowComponentsAssembling;
  outFlowComponentsAssembling << "flowComponentsAssemblingLevel" << pyramidLevel << ".txt";
  CCTagFileDebug::instance().newSession(outFlowComponentsAssembling.str());
  CCTagFileDebug::instance().initFlowComponentsIndex(2);
#endif

  BOOST_FOREACH(const Candidate & candidate, vCandidateLoopTwo)
  {
#ifdef CCTAG_SERIALIZE
    CCTagFileDebug::instance().resetFlowComponent();
    std::vector<Candidate> componentCandidates;
#ifdef DEBUG
    CCTagFileDebug::instance().resetFlowComponent();
#endif
#endif

    // todo@Lilian: remove copies -- find another solution
    std::vector<EdgePoint*> outerEllipsePoints = candidate._outerEllipsePoints;
    cctag::numerical::geometry::Ellipse outerEllipse = candidate._outerEllipse;
    std::vector<EdgePoint*> filteredChildrens = candidate._filteredChildrens;

    std::vector< std::vector< DirectedPoint2d<double> > > cctagPoints;

    try
    {
      double quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);

      if (params._searchForAnotherSegment)
      {
        if ((quality > 0.25) && (quality < 0.7))
        {
          // Search for another segment
          flowComponentAssembling( quality, candidate, vCandidateLoopTwo,
                  outerEllipse, outerEllipsePoints, cctagPoints, params
#ifndef CCTAG_SERIALIZE
                  );
#else
                  , componentCandidates);
#endif
        }
      }

      // Add the flowComponent from candidate to cctagPoints // Add intermediary points - required ? todo@Lilian
      // cctagPoints may be not empty, i.e. when the assembling succeed.
      if (! addCandidateFlowtoCCTag(candidate._filteredChildrens, 
              candidate._outerEllipsePoints, outerEllipse,
              cctagPoints, params._nCrowns * 2))
      {
        DO_TALK( CCTAG_COUT_DEBUG("Points outside the outer ellipse OR CCTag not valid : bad gradient orientations"); )
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTSOUTSIDE_OR_BADGRADORIENT);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        continue;
      }
      else
      {
#ifdef CCTAG_SERIALIZE
        componentCandidates.push_back(candidate);
#endif
        DO_TALK( CCTAG_COUT_DEBUG("Points inside the outer ellipse and good gradient orientations"); )
      }
      // Create ellipse with its real size from original image.
      cctag::numerical::geometry::Ellipse rescaleEllipse(outerEllipse.center(), outerEllipse.a() * scale, outerEllipse.b() * scale, outerEllipse.angle());
      
      int realPixelPerimeter = rasterizeEllipsePerimeter(rescaleEllipse);

      double realSizeOuterEllipsePoints = quality*realPixelPerimeter;

      // Naive reject condition todo@Lilian
      if ( ( ( quality <= 0.35 ) && ( realSizeOuterEllipsePoints >= 300.0 ) ) ||//0.35
               ( ( quality <= 0.45 ) && ( realSizeOuterEllipsePoints >= 200.0 ) && ( realSizeOuterEllipsePoints < 300.0 ) ) ||//0.45
               ( ( quality <= 0.50 ) && ( realSizeOuterEllipsePoints >= 100.0 ) && ( realSizeOuterEllipsePoints < 200.0 ) ) ||//0.50
               ( ( quality <= 0.50 ) && ( realSizeOuterEllipsePoints >= 70.0  ) && ( realSizeOuterEllipsePoints < 100.0 ) ) ||//0.5
               ( ( quality <= 0.96 ) && ( realSizeOuterEllipsePoints >= 50.0  ) && ( realSizeOuterEllipsePoints < 70.0 ) ) ||//0.96
               ( realSizeOuterEllipsePoints < 50.0  ) )
      {
              DO_TALK( CCTAG_COUT_DEBUG( "Not enough outer ellipse points: realSizeOuterEllipsePoints : " << realSizeOuterEllipsePoints << ", rasterizeEllipsePerimeter : " << rasterizeEllipsePerimeter( outerEllipse )*scale << ", quality : " << quality ); )
              continue;
      }

      cctag::Point2dN<double> markerCenter;
      cctag::numerical::BoundedMatrix3x3d markerHomography;

      const double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();

      if (ratioSemiAxes > 8.0 || ratioSemiAxes < 0.125)
      {
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RATIO_SEMIAXIS);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        DO_TALK( CCTAG_COUT_DEBUG("Too high ratio between semi-axes!"); )
        continue;
      }

      std::vector<double> vDistFinal;
      vDistFinal.clear();
      vDistFinal.reserve(outerEllipsePoints.size());

      double resSquare = 0;
      double distMax = 0;

      BOOST_FOREACH(EdgePoint * p, outerEllipsePoints)
      {
        double distFinal = numerical::distancePointEllipse(*p, outerEllipse, 1.0);
        resSquare += distFinal; //*distFinal;

        if (distFinal > distMax)
        {
          distMax = distFinal;
        }
      }

      resSquare = sqrt(resSquare);
      resSquare /= outerEllipsePoints.size();


      numerical::geometry::Ellipse qIn, qOut;
      computeHull(outerEllipse, 3.6, qIn, qOut);

      bool isValid = true;

      BOOST_FOREACH(const EdgePoint * p, outerEllipsePoints)
      {
        if (!isInHull(qIn, qOut, p))
        {
          isValid = false;
          break;
        }
      }
      if (!isValid)
      {
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTS_OUTSIDE_ELLHULL);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);

        DO_TALK( CCTAG_COUT_DEBUG("Distance max to high!"); )
        continue;
      }

      std::vector< Point2dN<int> > vPoint;

      double quality2 = 0;

      BOOST_FOREACH(const EdgePoint* p, outerEllipsePoints)
      {
        quality2 += p->_normGrad;
      }

      quality2 *= scale;

      markers.push_back(new CCTag(-1, outerEllipse.center(), cctagPoints,
              outerEllipse, markerHomography, pyramidLevel, scale, quality2));
#ifdef CCTAG_SERIALIZE
      markers.back().setFlowComponents(componentCandidates);
#ifdef DEBUG
      CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PASS_ALLTESTS);
      CCTagFileDebug::instance().incrementFlowComponentIndex(0);
#endif
#endif

      DO_TALK( CCTAG_COUT_DEBUG("------------------------------Added marker------------------------------"); )
    }
    catch (...)
    {
      CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RAISED_EXCEPTION);
      CCTagFileDebug::instance().incrementFlowComponentIndex(0);
      // Ellipse fitting don't pass.
      CCTAG_COUT_CURRENT_EXCEPTION;
      DO_TALK( CCTAG_COUT_DEBUG( "Exception raised" ); )
    }
  }

  boost::posix_time::ptime tstop2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d2 = tstop2 - tstop1;
  const double spendTime2 = d2.total_milliseconds();
  DO_TALK( CCTAG_COUT_OPTIM("Time in the 2nd loop" << spendTime2 << " ms"); )

  //	markers.sort();

  DO_TALK( CCTAG_COUT_DEBUG("Markers creation time: " << t3.elapsed()); )
  // POP_LEAVE;
}


void createImageForVoteResultDebug(
        const cv::Mat & src,
        const WinnerMap & winners,
        std::size_t nLevel)
{
#ifdef CCTAG_SERIALIZE 
  {
    std::size_t mx = 0;
    
    cv::Mat imgVote(src.rows, src.cols, CV_8UC1, cv::Scalar(0,0,0));

    for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr)
    {
      EdgePoint* winner = itr->first;
      std::list<EdgePoint*> v = itr->second;
      if (mx < v.size())
      {
        mx = v.size();
      }
    }

    for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr)
    {
      EdgePoint* winner = itr->first;
      std::list<EdgePoint*> v = itr->second;
      imgVote.at<uchar>(winner->y(),winner->x()) = (unsigned char) ((v.size() * 10.0));
    }

    //std::stringstream outFilenameVote;
    //outFilenameVote << "/home/lilian/data/vote_" << nLevel << ".png";
    //imwrite(outFilenameVote.str(), imgVote);
    
    std::stringstream outFilenameVote;
    outFilenameVote << "voteLevel" << CCTagVisualDebug::instance().getPyramidLevel();
    CCTagVisualDebug::instance().initBackgroundImage(imgVote);
    CCTagVisualDebug::instance().newSession(outFilenameVote.str());
  }
#endif
}

void cctagDetection(CCTag::List& markers,
        const std::size_t frame, 
        const cv::Mat & imgGraySrc,
        const Parameters & params,
        const cctag::CCTagMarkersBank & bank,
        const bool bDisplayEllipses)
{
  // POP_ENTER;
  using namespace cctag;
  using namespace boost::numeric::ublas;
  //using namespace boost::gil;

#ifdef CCTAG_OPTIM
  boost::posix_time::time_duration duration;
#endif // CCTAG_OPTIM
  
  std::srand(1);
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t00(boost::posix_time::microsec_clock::local_time());
#endif // CCTAG_OPTIM
  
  ImagePyramid imagePyramid(imgGraySrc.cols, imgGraySrc.rows, params._numberOfProcessedMultiresLayers);

  popart::TagPipe* pipe1 = new popart::TagPipe;
    
  uint32_t w = imgGraySrc.size().width;
  uint32_t h = imgGraySrc.size().height;
  pipe1->initialize( w, h, params );

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t01(boost::posix_time::microsec_clock::local_time());
  duration = t01 - t00;
  CCTAG_COUT_OPTIM("Time to init GPU & CPU: " << duration.total_milliseconds() << " ms");
#endif // CCTAG_OPTIM

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t10(boost::posix_time::microsec_clock::local_time());
#endif // CCTAG_OPTIM
  // unsigned char* pix = frame.data;
  assert( imgGraySrc.elemSize() == 1 );
  assert( imgGraySrc.isContinuous() );
  assert( imgGraySrc.type() == CV_8U );
  unsigned char* pix = imgGraySrc.data;

  pipe1->load( pix );
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t11(boost::posix_time::microsec_clock::local_time());
  duration = t11 - t10;
  CCTAG_COUT_OPTIM("Time to load img into GPU: " << duration.total_milliseconds() << " ms");
#endif // CCTAG_OPTIM

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t20(boost::posix_time::microsec_clock::local_time());
#endif // CCTAG_OPTIM
  pipe1->tagframe( params ); // pix, w, h, params );
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t21(boost::posix_time::microsec_clock::local_time());
  duration = t21 - t20;
  CCTAG_COUT_OPTIM("Time for GPU processing: " << duration.total_milliseconds() << " ms");
#endif // CCTAG_OPTIM

#ifndef NDEBUG
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t03(boost::posix_time::microsec_clock::local_time());
#endif // CCTAG_OPTIM
  pipe1->debug( pix, params );
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t13(boost::posix_time::microsec_clock::local_time());
  duration = t13 - t03;
  CCTAG_COUT_OPTIM("Time in GPU debug: " << duration.total_milliseconds() << " ms");
#endif // CCTAG_OPTIM
#endif // not NDEBUG
  
  cctagMultiresDetection( markers, imgGraySrc, imagePyramid, frame, pipe1, params );

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  duration = t1 - t00;
  DO_TALK( CCTAG_COUT_OPTIM("TIME IN DETECTION: " << duration.total_milliseconds() << " ms"); )
#endif
  
  CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());
  // Identification step
  if (params._doIdentification)
  {
    CCTag::List::iterator it = markers.begin();
    while (it != markers.end())
    {
      CCTag & cctag = *it;

      const int detected = cctag::identification::identify(
              cctag,
              bank.getMarkers(),
              imagePyramid.getLevel(0)->getSrc(),
              params);
      
      cctag.setStatus(detected);
      ++it;
    }
#ifdef CCTAG_OPTIM
      boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
      duration = t2 - t1;
      const double spendTime2 = duration.total_milliseconds();
      CCTAG_COUT_OPTIM("TIME IN IDENTIFICATION: " << spendTime2 << " ms");
#endif
  }
  
  markers.sort();

  CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());
  CCTagVisualDebug::instance().writeIdentificationView(markers);
  CCTagFileDebug::instance().newSession("identification.txt");

  BOOST_FOREACH(const CCTag & marker, markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }

  // POP_LEAVE;
}

} // namespace cctag
