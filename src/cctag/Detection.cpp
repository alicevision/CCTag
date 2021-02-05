/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/FileDebug.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/Detection.hpp>
#include <cctag/Vote.hpp>
#include <cctag/utils/VisualDebug.hpp>
#include <cctag/Multiresolution.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/CCTagFlowComponent.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/Identification.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/Types.hpp>
#include <cctag/Canny.hpp>
#include <cctag/utils/Defines.hpp>
#include <cctag/utils/Talk.hpp> // for DO_TALK macro
#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif

#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/unordered/unordered_set.hpp>
#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <exception>
#include <fstream>
#include <sstream>
#include <list>
#include <utility>
#include <memory>
#ifdef CCTAG_WITH_CUDA
#include <cctag/cuda/cctag_cuda_runtime.h> // only for debugging
#endif // CCTAG_WITH_CUDA

#include <tbb/tbb.h>

using namespace std;

namespace cctag
{

namespace { using CandidatePtr = std::unique_ptr<Candidate>; }

/* These are the CUDA pipelines that we instantiate for parallel processing.
 * We need at least one.
 * It is uncertain whether the CPU code can handle parallel pipe, but the CUDA
 * code should be able to.
 * BEWARE: this is untested
 */
std::vector<cctag::TagPipe*> cudaPipelines;

static void constructFlowComponentFromSeed(
        EdgePoint * seed,
        EdgePointCollection& edgeCollection,
        std::vector<CandidatePtr> & vCandidateLoopOne,
        const Parameters & params)
{
  static tbb::mutex G_SortMutex;
  
  assert( seed );
  // Check if the seed has already been processed, i.e. belongs to an already
  // reconstructed flow component.
  if (!edgeCollection.test_processed_in(seed))
  {

    CandidatePtr candidate(new Candidate);

    candidate->_seed = seed;
    std::list<EdgePoint*> & convexEdgeSegment = candidate->_convexEdgeSegment;

    // Convex edge linking from the seed in both directions. The linking
    // is performed until the convexity is lost.
    edgeLinking(edgeCollection, convexEdgeSegment, seed,
            params._windowSizeOnInnerEllipticSegment, params._averageVoteMin);

    // Compute the average number of received points.
    int nReceivedVote = 0;
    int nVotedPoints = 0;

    for (EdgePoint* p : convexEdgeSegment)
    {
      auto votersSize = edgeCollection.voters_size(p);
      nReceivedVote += votersSize;
      if (votersSize > 0)
        ++nVotedPoints;
    }
    
    {
      tbb::mutex::scoped_lock lock(G_SortMutex);
      candidate->_averageReceivedVote = (float) (nReceivedVote*nReceivedVote) / (float) nVotedPoints;
      auto it = std::lower_bound(vCandidateLoopOne.begin(), vCandidateLoopOne.end(), candidate,
        [](const CandidatePtr& c1, const CandidatePtr& c2) { return c1->_averageReceivedVote > c2->_averageReceivedVote; });
      vCandidateLoopOne.insert(it, std::move(candidate));
    }
  }
}

static void completeFlowComponent(
  Candidate & candidate,
  const EdgePointCollection& edgeCollection,
  std::vector<Candidate> & vCandidateLoopTwo,
  std::size_t& nSegmentOut,
  std::size_t runId,
  const Parameters & params)
{
  static tbb::spin_mutex G_UpdateMutex;
  static tbb::mutex G_InsertMutex;
  
  try
  {
    std::list<EdgePoint*> children;

    childrenOf(edgeCollection, candidate._convexEdgeSegment, children);

    if (children.size() < params._minPointsSegmentCandidate)
    {
      return;
    }

    candidate._score = children.size();

    float SmFinal = 1e+10;

    std::vector<EdgePoint*> & filteredChildren = candidate._filteredChildren;

    outlierRemoval(
            children,
            filteredChildren,
            SmFinal, 
            params._threshRobustEstimationOfOuterEllipse,
            kWeight,
            60);

    if (filteredChildren.size() < 5)
    {
      DO_TALK( CCTAG_COUT_DEBUG(" filteredChildren.size() < 5 "); )
      return;
    }

    std::size_t nLabel = -1;

    {
      ssize_t nSegmentCommon = -1;

      for(EdgePoint * p : filteredChildren)
      {
        if (p->_nSegmentOut != -1)
        {
          nSegmentCommon = p->_nSegmentOut;
          break;
        }
      }
      

      if (nSegmentCommon == -1)
      {
        {
          tbb::spin_mutex::scoped_lock lock(G_UpdateMutex);
          nLabel = nSegmentOut;
          ++nSegmentOut;
        }
      }
      else
      {
        nLabel = nSegmentCommon;
      }

      for(EdgePoint * p : filteredChildren)
      {
        p->_nSegmentOut = nLabel;
      }
    }

    std::vector<EdgePoint*> & outerEllipsePoints = candidate._outerEllipsePoints;
    cctag::numerical::geometry::Ellipse & outerEllipse = candidate._outerEllipse;

    bool goodInit = false;

    goodInit = ellipseGrowingInit(filteredChildren, outerEllipse);

    ellipseGrowing2(edgeCollection, filteredChildren, outerEllipsePoints, outerEllipse,
                    params._ellipseGrowingEllipticHullWidth, runId, goodInit);

    candidate._nLabel = nLabel;

    std::vector<float> vDistFinal;
    vDistFinal.clear();
    vDistFinal.reserve(outerEllipsePoints.size());

    float distMax = 0;

    for(EdgePoint * p : outerEllipsePoints)
    {
      float distFinal = numerical::distancePointEllipse(*p, outerEllipse);
      vDistFinal.push_back(distFinal);

      if (distFinal > distMax)
      {
        distMax = distFinal;
      }
    }

    SmFinal = numerical::medianRef(vDistFinal);

    if (SmFinal > params._thrMedianDistanceEllipse)
    {
      DO_TALK( CCTAG_COUT_DEBUG("SmFinal < params._thrMedianDistanceEllipse -- after ellipseGrowing"); )
      return;
    }

    float quality = (float) outerEllipsePoints.size() / (float) rasterizeEllipsePerimeter(outerEllipse);
    if (quality > 1.1)
    {
      DO_TALK( CCTAG_COUT_DEBUG("Quality too high!"); )
      return;
    }

    float ratioSemiAxes = outerEllipse.a() / outerEllipse.b();
    if ((ratioSemiAxes < 0.05) || (ratioSemiAxes > 20))
    {
      DO_TALK( CCTAG_COUT_DEBUG("Too high semi-axis ratio!"); )
      return;
    }

    {
      tbb::mutex::scoped_lock lock(G_InsertMutex);
      vCandidateLoopTwo.push_back(candidate);
    }

#ifdef CCTAG_SERIALIZE
    // Add children to output the filtering results (from outlierRemoval)
    vCandidateLoopTwo.back().setchildren(children);

    // Write all selectedFlowComponent
    CCTagFlowComponent flowComponent(edgeCollection, outerEllipsePoints, children, filteredChildren,
                                     outerEllipse, candidate._convexEdgeSegment,
                                    *(candidate._seed), params._nCircles);
    CCTagFileDebug::instance().outputFlowComponentInfos(flowComponent);
#endif

  }
  catch (cv::Exception& e)
  {
    DO_TALK( CCTAG_COUT_DEBUG( "OpenCV exception: " +  e.what() ); )
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
static void flowComponentAssembling(
        EdgePointCollection& edgeCollection,
        float & quality,
        const Candidate & candidate,
        const std::vector<Candidate> & vCandidateLoopTwo,
        numerical::geometry::Ellipse & outerEllipse,
        std::vector<EdgePoint*>& outerEllipsePoints,
        std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > >& cctagPoints,
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

  float ratioExpension = 2.5;
  numerical::geometry::Circle circularResearchArea(
         Point2d<Eigen::Vector3f>( candidate._seed->x(), candidate._seed->y() ),
         candidate._seed->_flowLength * ratioExpension);

  {
    int i = 0;
    // Search for another segment
    for(const Candidate & anotherCandidate : vCandidateLoopTwo)
    {
      if (&candidate != &anotherCandidate)
      {
        if (candidate._nLabel != anotherCandidate._nLabel)
        {
          if ((anotherCandidate._seed->_flowLength / candidate._seed->_flowLength > 0.666)
                  && (anotherCandidate._seed->_flowLength / candidate._seed->_flowLength < 1.5))
          {
            if (isInEllipse(circularResearchArea, 
                    cctag::Point2d<Eigen::Vector3f>(float(anotherCandidate._seed->x()), float(anotherCandidate._seed->y()))))
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
  }

  DO_TALK( CCTAG_COUT_VAR_DEBUG(iMax); )
  DO_TALK( CCTAG_COUT_VAR_DEBUG(*(vCandidateLoopTwo[iMax])._seed); )

  if (score > 0)
  {
    const Candidate & selectedCandidate = vCandidateLoopTwo[iMax];
    DO_TALK( CCTAG_COUT_VAR_DEBUG(selectedCandidate._outerEllipse); )

    if( isAnotherSegment(edgeCollection, outerEllipse, outerEllipsePoints, 
            selectedCandidate._filteredChildren, selectedCandidate,
            cctagPoints, params._nCrowns * 2,
            params._thrMedianDistanceEllipse) )
    {
      quality = (float) outerEllipsePoints.size() / (float) rasterizeEllipsePerimeter(outerEllipse);

#ifdef CCTAG_SERIALIZE
      componentCandidates.push_back(selectedCandidate);
#endif
      CCTagFileDebug::instance().setFlowComponentAssemblingState(true, iMax);
    }
  }

  boost::posix_time::ptime tstop(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d = tstop - tstart;
  const float spendTime = d.total_milliseconds();
}

static void cctagDetectionFromEdgesLoopTwoIteration(
  CCTag::List& markers,
  EdgePointCollection& edgeCollection,
  const std::vector<Candidate>& vCandidateLoopTwo,
  size_t iCandidate,
  int pyramidLevel,
  float scale,
  const Parameters& params)
{
    static tbb::mutex G_InsertMutex;
    
    const Candidate& candidate = vCandidateLoopTwo[iCandidate];

#ifdef CCTAG_SERIALIZE
    CCTagFileDebug::instance().resetFlowComponent();
    std::vector<Candidate> componentCandidates;
#ifdef DEBUG
    CCTagFileDebug::instance().resetFlowComponent();
#endif
#endif

    // TODO@stian: remove copying
    std::vector<EdgePoint*> outerEllipsePoints = candidate._outerEllipsePoints;
    cctag::numerical::geometry::Ellipse outerEllipse = candidate._outerEllipse;

    std::vector< std::vector< DirectedPoint2d<Eigen::Vector3f> > > cctagPoints;

    // todo@Lilian: The following block along with its called function are ugly:
    // flowComponentAssembling should be performed from the connected edges in
    // downsample images.
    try
    {
      float quality = (float) outerEllipsePoints.size() / (float) rasterizeEllipsePerimeter(outerEllipse);

      if (params._searchForAnotherSegment)
      {
        if ((quality > 0.25) && (quality < 0.7))
        {
          // Search for another segment
          flowComponentAssembling( edgeCollection, quality, candidate, vCandidateLoopTwo,
                  outerEllipse, outerEllipsePoints, cctagPoints, params
#ifndef CCTAG_SERIALIZE
                  );
#else
                  , componentCandidates);
#endif
        }
      }

      // Add the flowComponent from candidate to cctagPoints
      if (! addCandidateFlowtoCCTag(edgeCollection, candidate._filteredChildren,
              candidate._outerEllipsePoints, outerEllipse,
              cctagPoints, params._nCrowns * 2))
      {
        DO_TALK( CCTAG_COUT_DEBUG("Points outside the outer ellipse OR CCTag not valid : bad gradient orientations"); )
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTSOUTSIDE_OR_BADGRADORIENT);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        return;
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

      float realSizeOuterEllipsePoints = quality*realPixelPerimeter;

      // todo@Lilian: remove this heuristic
      if ( ( ( quality <= 0.35 ) && ( realSizeOuterEllipsePoints >= 300.0 ) ) ||
               ( ( quality <= 0.45f ) && ( realSizeOuterEllipsePoints >= 200.0 ) && ( realSizeOuterEllipsePoints < 300.0 ) ) ||
               ( ( quality <= 0.5f ) && ( realSizeOuterEllipsePoints >= 100.0 ) && ( realSizeOuterEllipsePoints < 200.0 ) ) ||
               ( ( quality <= 0.5f ) && ( realSizeOuterEllipsePoints >= 70.0  ) && ( realSizeOuterEllipsePoints < 100.0 ) ) ||
               ( ( quality <= 0.96f ) && ( realSizeOuterEllipsePoints >= 50.0  ) && ( realSizeOuterEllipsePoints < 70.0 ) ) ||
               ( realSizeOuterEllipsePoints < 50.0  ) )
      {
              DO_TALK( CCTAG_COUT_DEBUG( "Not enough outer ellipse points: realSizeOuterEllipsePoints : " << realSizeOuterEllipsePoints << ", rasterizeEllipsePerimeter : " << rasterizeEllipsePerimeter( outerEllipse )*scale << ", quality : " << quality ); )
              return;
      }

      cctag::Point2d<Eigen::Vector3f> markerCenter;
      Eigen::Matrix3f markerHomography = Eigen::Matrix3f::Zero();

      const float ratioSemiAxes = outerEllipse.a() / outerEllipse.b();

      if (ratioSemiAxes > 8.0 || ratioSemiAxes < 0.125)
      {
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RATIO_SEMIAXIS);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        DO_TALK( CCTAG_COUT_DEBUG("Too high ratio between semi-axes!"); )
        return;
      }

      // TODO@stian: remove allocation from loop iteration
      std::vector<float> vDistFinal;
      vDistFinal.clear();
      vDistFinal.reserve(outerEllipsePoints.size());

      float resSquare = 0;
      float distMax = 0;

      // TODO@stian: TBB parallel reduction
      for(EdgePoint * p : outerEllipsePoints)
      {
        float distFinal = numerical::distancePointEllipse(*p, outerEllipse);
        resSquare += distFinal; //*distFinal;

        if (distFinal > distMax)
        {
          distMax = distFinal;
        }
      }

      resSquare = sqrt(resSquare);
      resSquare /= outerEllipsePoints.size();

      numerical::geometry::Ellipse qIn, qOut;
      computeHull(outerEllipse, 3.6f, qIn, qOut);

      bool isValid = true;

      for(const EdgePoint * p : outerEllipsePoints)
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
        return;
      }

      std::vector< Point2d<Eigen::Vector3i> > vPoint;

      float quality2 = 0;

      // todo@Lilian: no longer used ?
      for(const EdgePoint* p : outerEllipsePoints)
      {
        quality2 += p->normGradient();
      }
      
      quality2 *= scale;

      CCTag* tag = new CCTag( -1,
                              outerEllipse.center(),
                              cctagPoints,
                              outerEllipse,
                              markerHomography,
                              pyramidLevel,
                              scale,
                              quality2 );
#ifdef CCTAG_SERIALIZE
      tag->setFlowComponents( componentCandidates, edgeCollection);
#endif
      
      {
        tbb::mutex::scoped_lock lock(G_InsertMutex);
        markers.push_back( tag ); // markers takes responsibility for delete
      }
#ifdef CCTAG_SERIALIZE
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
      //CCTAG_COUT_CURRENT_EXCEPTION;
      DO_TALK( CCTAG_COUT_DEBUG( "Exception raised" ); )
    }
}

void cctagDetectionFromEdges(
        CCTag::List&            markers,
        EdgePointCollection& edgeCollection,
        const cv::Mat&          src,
        const std::vector<EdgePoint*>& seeds,
        std::size_t frame,
        int pyramidLevel,
        float scale,
        const Parameters & providedParams,
        cctag::logtime::Mgmt* durations )
{
  const Parameters& params = Parameters::OverrideLoaded ?
    Parameters::Override : providedParams;

  // Call for debug only. Write the vote result as an image.
  createImageForVoteResultDebug(src, pyramidLevel);

  // Set some timers
  boost::timer::cpu_timer t3;
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

  std::vector<CandidatePtr> vCandidateLoopOne;

  // Process all the first-nSeedsToProcess seeds.
  // In the following loop, a seed will lead to a flow component if it lies
  // on the inner ellipse of a CCTag.
  // The edge points lying on the inner ellipse and their voters (lying on the outer ellipse)
  // will be collected and constitute the initial data of a flow component.
  
#ifndef CCTAG_SERIALIZE
  tbb::parallel_for(size_t(0), nSeedsToProcess, [&](int iSeed) {
#else 
  for(size_t iSeed=0 ; iSeed < nSeedsToProcess; ++iSeed)
  {
#endif
    assert( seeds[iSeed] );
    constructFlowComponentFromSeed(seeds[iSeed], edgeCollection, vCandidateLoopOne, params);
#ifndef CCTAG_SERIALIZE
  });
#else
  }
#endif

  const std::size_t nFlowComponentToProcessLoopTwo = 
          std::min(vCandidateLoopOne.size(), params._maximumNbCandidatesLoopTwo);

  std::vector<Candidate> vCandidateLoopTwo;
  vCandidateLoopTwo.reserve(nFlowComponentToProcessLoopTwo);

  // Second main loop:
  // From the flow components selected in the first loop, the outer ellipse will
  // be here entirely recovered.
  // The GPU implementation should stop at this point => layers ->  EdgePoint* creation.

  CCTagVisualDebug::instance().initBackgroundImage(src);
  CCTagVisualDebug::instance().newSession( "completeFlowComponent" );
  
#ifndef CCTAG_SERIALIZE
  tbb::parallel_for(size_t(0), nFlowComponentToProcessLoopTwo, [&](size_t iCandidate) {
#else
    for(size_t iCandidate=0 ; iCandidate < nFlowComponentToProcessLoopTwo; ++iCandidate)
    {
#endif
      size_t runId = iCandidate;
      completeFlowComponent(*vCandidateLoopOne[iCandidate], edgeCollection, vCandidateLoopTwo, nSegmentOut, runId, params);
#ifndef CCTAG_SERIALIZE  
    });
#else
  }
#endif
  
  DO_TALK(
    CCTAG_COUT_VAR_DEBUG(vCandidateLoopTwo.size());
    CCTAG_COUT_DEBUG("================= List of seeds =================");
    for(const Candidate & anotherCandidate : vCandidateLoopTwo)
    {
      CCTAG_COUT_DEBUG("X = [ " << anotherCandidate._seed->x() << " , " << anotherCandidate._seed->y() << "]");
    }
  )

  boost::posix_time::ptime tstop1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d1 = tstop1 - tstart0;
  const float spendTime1 = d1.total_milliseconds();

#if defined CCTAG_SERIALIZE && defined DEBUG
  std::stringstream outFlowComponentsAssembling;
  outFlowComponentsAssembling << "flowComponentsAssemblingLevel" << pyramidLevel << ".txt";
  CCTagFileDebug::instance().newSession(outFlowComponentsAssembling.str());
  CCTagFileDebug::instance().initFlowComponentsIndex(2);
#endif

  const size_t candidateLoopTwoCount = vCandidateLoopTwo.size();

#ifndef CCTAG_SERIALIZE
  tbb::parallel_for(size_t(0), candidateLoopTwoCount, [&](size_t iCandidate) {
#else
  for(size_t iCandidate=0 ; iCandidate < vCandidateLoopTwo.size(); ++iCandidate)
#endif
    cctagDetectionFromEdgesLoopTwoIteration(markers, edgeCollection, vCandidateLoopTwo, iCandidate,
      pyramidLevel, scale, params);
#ifndef CCTAG_SERIALIZE
  });
#endif
  
  boost::posix_time::ptime tstop2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d2 = tstop2 - tstop1;
  const float spendTime2 = d2.total_milliseconds();
}


void createImageForVoteResultDebug(
        const cv::Mat & src,
        std::size_t nLevel)
{
#if defined(CCTAG_SERIALIZE) && 0 // todo@lilian: fixme
  {
    std::size_t mx = 0;
    
    cv::Mat imgVote(src.rows, src.cols, CV_8UC1, cv::Scalar(0,0,0));

    for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr)
    {
      EdgePoint* winner = itr->first;
      const std::vector<EdgePoint*>& v = itr->second;
      if (mx < v.size())
      {
        mx = v.size();
      }
    }

    for (WinnerMap::const_iterator itr = winners.begin(); itr != winners.end(); ++itr)
    {
      EdgePoint* winner = itr->first;
      const std::vector<EdgePoint*>& v = itr->second;
      imgVote.at<uchar>(winner->y(),winner->x()) = (unsigned char) ((v.size() * 10.0));
    }
    
    std::stringstream outFilenameVote;
    outFilenameVote << "voteLevel" << CCTagVisualDebug::instance().getPyramidLevel();
    CCTagVisualDebug::instance().initBackgroundImage(imgVote);
    CCTagVisualDebug::instance().newSession(outFilenameVote.str());
  }
#endif
}

#ifdef CCTAG_WITH_CUDA
cctag::TagPipe* initCuda( int      pipeId,
                          uint32_t width,
                          uint32_t height, 
                          const Parameters & params,
                          cctag::logtime::Mgmt* durations )
{
    PinnedCounters::setGlobalMax( params._pinnedCounters,
                                  params._pinnedNearbyPoints );

    if( cudaPipelines.size() <= pipeId )
    {
        cudaPipelines.resize( pipeId+1 );
    }

    cctag::TagPipe* pipe1 = cudaPipelines[pipeId];

    if( ! pipe1 ) {
        pipe1 = new cctag::TagPipe( params );
        pipe1->initialize( width, height, durations );
        cudaPipelines[pipeId] = pipe1;
    } else {
        if( width  != pipe1->getWidth(0) ||
            height != pipe1->getHeight(0) ) {
            std::cerr << "We cannot change the input frame resolution (yet)" << std::endl;
            exit( -1 );
        }
    }
    return pipe1;
}
#endif // CCTAG_WITH_CUDA

/**
 * @brief Perform the CCTag detection on a gray scale image
 * 
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are valid ones. (status available via getStatus()) 
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] imgGraySrc Gray scale input image.
 * @param[in] providedParams Contains all the parameters.
 * @param[in] bank CCTag bank.
 * @param[in] No longer used.
 */
void cctagDetection(
        CCTag::List& markers,
        int          pipeId,
        std::size_t frame,
        const cv::Mat & imgGraySrc,
        const Parameters & providedParams,
        const cctag::CCTagMarkersBank & bank,
        bool bDisplayEllipses,
        cctag::logtime::Mgmt* durations )

{
    using namespace cctag;
    
    const Parameters& params = Parameters::OverrideLoaded ?
      Parameters::Override : providedParams;

    if( durations ) durations->log( "start" );
  
    std::srand(1);

#ifdef CCTAG_WITH_CUDA
    bool cuda_allocates = params._useCuda;
#else
    bool cuda_allocates = false;
#endif
  
    ImagePyramid imagePyramid( imgGraySrc.cols,
                               imgGraySrc.rows,
                               params._numberOfProcessedMultiresLayers,
                               cuda_allocates );

    cctag::TagPipe* pipe1 = nullptr;
#ifdef CCTAG_WITH_CUDA
    if( params._useCuda ) {
        pipe1 = initCuda( pipeId,
                          imgGraySrc.size().width,
	                      imgGraySrc.size().height,
	                      params,
	                      durations );

        if( durations ) durations->log( "after initCuda" );

        assert( imgGraySrc.elemSize() == 1 );
        assert( imgGraySrc.isContinuous() );
        assert( imgGraySrc.type() == CV_8U );
        unsigned char* pix = imgGraySrc.data;

        pipe1->load( frame, pix );

        if( durations ) {
            cudaDeviceSynchronize();
            durations->log( "after CUDA load" );
        }

        pipe1->tagframe( );

        if( durations ) durations->log( "after CUDA stages" );
    } else { // not params.useCuda
#endif // CCTAG_WITH_CUDA

        imagePyramid.build( imgGraySrc,
                            params._cannyThrLow,
                            params._cannyThrHigh,
                            &params );

#ifdef CCTAG_WITH_CUDA
    } // not params.useCuda
#endif // CCTAG_WITH_CUDA
  
    if( durations ) durations->log( "before cctagMultiresDetection" );

    cctagMultiresDetection( markers,
                            imgGraySrc,
                            imagePyramid,
                            frame,
                            pipe1,
                            params,
                            durations );

    if( durations ) durations->log( "after cctagMultiresDetection" );

#ifdef CCTAG_WITH_CUDA
    if( pipe1 ) {
        /* identification in CUDA requires a host-side nearby point struct
         * in pinned memory for safe, non-blocking memcpy.
         */
        if( markers.size() > MAX_MARKER_FOR_IDENT ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
              << "   Found more than " << MAX_MARKER_FOR_IDENT << " (" << markers.size() << ") markers" << endl;
        }

        for( CCTag& tag : markers ) {
            tag.acquireNearbyPointMemory( pipe1->getId() );
        }
    }
#endif // CCTAG_WITH_CUDA
  
    CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());

    // Identification step
    if (params._doIdentification)
    {
      CCTagVisualDebug::instance().resetMarkerIndex();

        const std::size_t numTags  = markers.size();

#ifdef CCTAG_WITH_CUDA
        if( pipe1 && numTags > 0 ) {
            pipe1->checkTagAllocations( numTags, params );
        }
#endif // CCTAG_WITH_CUDA

        std::vector<std::vector<cctag::ImageCut> > vSelectedCuts( numTags );
		std::vector<int> detected(numTags, -1);
        int                          tagIndex = 0;

        for( const CCTag& cctag : markers ) {
            detected[tagIndex] = cctag::identification::identify_step_1(
                tagIndex,
                cctag,
                vSelectedCuts[tagIndex],
                imagePyramid.getLevel(0)->getSrc(),
                params );

            tagIndex++;
        }

        if( markers.size() != numTags ) {
            cerr << __FILE__ << ":" << __LINE__ << " Number of markers has changed in identify_step_1" << endl;
        }

#ifdef CCTAG_WITH_CUDA
        if( pipe1 && numTags > 0 ) {
            pipe1->uploadCuts( numTags, &vSelectedCuts[0], params );

            tagIndex = 0;
            int debug_num_calls = 0;
            for( CCTag& cctag : markers ) {
                if( vSelectedCuts[tagIndex].size() <= 2 ) {
                    detected[tagIndex] = status::no_selected_cuts;
                } else if( detected[tagIndex] == status::id_reliable ) {
                    if( debug_num_calls >= numTags ) {
                        cerr << __FILE__ << ":" << __LINE__ << " center finding for more loops (" << debug_num_calls << ") than uploaded (" << numTags << ")?" << endl;
                    }
                    cctag::NearbyPoint* nearbyPointBuffer = cctag.getNearbyPointBuffer();
                    if(!nearbyPointBuffer)
                    {
                        detected[tagIndex] = status::no_selected_cuts;
                    }
                    else
                    {
                        pipe1->imageCenterOptLoop(
                            tagIndex,
                            numTags, // for debugging only
                            cctag.rescaledOuterEllipse(),
                            cctag.centerImg(),
                            vSelectedCuts[tagIndex].size(),
                            params,
                            nearbyPointBuffer );
                    }
                }

                tagIndex++;
            }
            cudaDeviceSynchronize();
        }
#endif // CCTAG_WITH_CUDA

        tagIndex = 0;

        CCTag::List::iterator it = markers.begin();
        while (it != markers.end())
        {
            CCTag & cctag = *it;

            if( detected[tagIndex] == status::id_reliable ) {
                detected[tagIndex] = cctag::identification::identify_step_2(
                    tagIndex,
                    cctag,
                    vSelectedCuts[tagIndex],
                    bank.getMarkers(),
                    imagePyramid.getLevel(0)->getSrc(),
                    pipe1,
                    params );
            }

            cctag.setStatus( detected[tagIndex] );
            ++it;

            tagIndex++;
        }
        if( durations ) durations->log( "after cctag::identification::identify" );
    }

#ifdef CCTAG_WITH_CUDA
    if( pipe1 ) {
        /* Releasing all points in all threads in the process.
         */
        CCTag::releaseNearbyPointMemory( pipe1->getId() );
    }
#endif
    
    // Delete overlapping markers while keeping the best ones.
    CCTag::List markersPrelim, markersFinal;
    for(const CCTag & marker : markers)
    {
        update(markersPrelim, marker);
    }

    for(const CCTag & marker : markersPrelim)
    {
      update(markersFinal, marker);
    }

    markers = markersFinal;
  
    markers.sort();

    CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());
    CCTagVisualDebug::instance().writeIdentificationView(markers);
    CCTagFileDebug::instance().newSession("identification.txt");

    for(const CCTag & marker : markers)
    {
        CCTagFileDebug::instance().outputMarkerInfos(marker);
    }
}

} // namespace cctag
