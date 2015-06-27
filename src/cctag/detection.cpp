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
#include <cctag/fileDebug.hpp>
#ifdef WITH_CUDA
  #include "cuda/tag.h"
#endif // WITH_CUDA

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
        Candidate & candidate, WinnerMap & winners,
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
        CCTAG_COUT_DEBUG(" childrens.size() < minPointsSegmentCandidate ");
        return;
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
        CCTAG_COUT_DEBUG(" filteredChildrens.size() < 5 ");
        return;
      }

      std::size_t nLabel = -1;

      {
        std::size_t nSegmentCommon = -1;

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
        CCTAG_COUT_DEBUG("SmFinal < params._thrMedianDistanceEllipse -- after ellipseGrowing");
        return;
      }

      double quality = (double) outerEllipsePoints.size() / (double) rasterizeEllipsePerimeter(outerEllipse);
      if (quality > 1.1)
      {
        CCTAG_COUT_DEBUG("Quality too high!");
        return;
      }

      double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();
      if ((ratioSemiAxes < 0.05) || (ratioSemiAxes > 20))
      {
        CCTAG_COUT_DEBUG("Too high ratio between semi-axes!");
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
      CCTAG_COUT_DEBUG( "OpenCV exception" );
#ifdef WITH_CUDA
      e.what();
#else
      const char* err_msg = e.what();
#endif
    }
  }
  catch (...)
  {
    CCTAG_COUT_DEBUG( "Exception raised in the second main loop." );
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
        std::vector< std::vector< Point2dN<double> > >& cctagPoints,
        const Parameters & params
#ifndef CCTAG_SERIALIZE
        )
#else
        , std::vector<Candidate> & componentCandidates)
#endif
{
  boost::posix_time::ptime tstart(boost::posix_time::microsec_clock::local_time());
  
  CCTAG_COUT_DEBUG("================= Look for another segment ==================");

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

  CCTAG_COUT_VAR_DEBUG(iMax);
  CCTAG_COUT_VAR_DEBUG(*(vCandidateLoopTwo[iMax])._seed);

  if (score > 0)
  {
    const Candidate & selectedCandidate = vCandidateLoopTwo[iMax];
    CCTAG_COUT_VAR_DEBUG(selectedCandidate._outerEllipse);

    if( isAnotherSegment(outerEllipse, outerEllipsePoints, 
            selectedCandidate._filteredChildrens, selectedCandidate,
            cctagPoints, params._numCrowns * 2,
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
#ifdef WITH_CUDA
  d.total_milliseconds();
#else
  const double spendTime = d.total_milliseconds();
#endif
}


void cctagDetectionFromEdges(
        CCTag::List& markers,
        std::vector<EdgePoint>& points,
        const boost::gil::gray8_view_t & sourceView,
        const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX,
        const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
        const EdgePointsImage& edgesMap,
        const std::size_t frame,
        int pyramidLevel,
        double scale,
        const Parameters & params)
{
  POP_ENTER;
  using namespace boost::gil;

  // Get vote winners
  WinnerMap winners;
  std::vector<EdgePoint*> seeds;

  // Voting procedure applied on every edge points.
  vote(points, seeds, edgesMap, winners, cannyGradX, cannyGradY, params);

  // Call for debug only. Write the vote result as an image.
  createImageForVoteResultDebug(sourceView, winners);

  // Set some timers
  boost::timer t3;
  boost::posix_time::ptime tstart0(boost::posix_time::microsec_clock::local_time());

  std::size_t nSegmentOut = 0;

#ifdef CCTAG_SERIALIZE
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
  // Sort the seeds based on the number of received votes.
  std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);

  const std::size_t nSeedsToProcess = std::min(seeds.size(), params._maximumNbSeeds);

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

  const std::size_t nCandidatesLoopOneToProcess = 
          std::min(vCandidateLoopOne.size(), params._maximumNbCandidatesLoopTwo);

  std::vector<Candidate> vCandidateLoopTwo;
  vCandidateLoopTwo.reserve(nCandidatesLoopOneToProcess);

  std::list<Candidate>::iterator it = vCandidateLoopOne.begin();
  std::size_t iCandidate = 0;

  // Second main loop:
  // From the flow components selected in the first loop, the outer ellipse will
  // be here entirely recovered.
  // The GPU implementation should stop at this point => layers ->  EdgePoint* creation.

  while (iCandidate < nCandidatesLoopOneToProcess)
  {
    completeFlowComponent(*it, winners, points, edgesMap, vCandidateLoopTwo, nSegmentOut, params);
    ++it;
    ++iCandidate;
  }

  CCTAG_COUT_VAR_DEBUG(vCandidateLoopTwo.size());
  CCTAG_COUT_DEBUG("================= List of seeds =================");
  BOOST_FOREACH(const Candidate & anotherCandidate, vCandidateLoopTwo)
  {
    CCTAG_COUT_DEBUG("X = [ " << anotherCandidate._seed->x() << " , " << anotherCandidate._seed->y() << "]");
  }

  boost::posix_time::ptime tstop1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d1 = tstop1 - tstart0;
  const double spendTime1 = d1.total_milliseconds();
  CCTAG_COUT_OPTIM(" Time in the 1st loop " << spendTime1 << " ms");

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

    std::vector< std::vector< Point2dN<double> > > cctagPoints;

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
              cctagPoints, params._numCrowns * 2))
      {
        CCTAG_COUT_DEBUG("Points outside the outer ellipse OR CCTag not valid : bad gradient orientations");
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTSOUTSIDE_OR_BADGRADORIENT);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        continue;
      }
      else
      {
#ifdef CCTAG_SERIALIZE
        componentCandidates.push_back(candidate);
#endif
        CCTAG_COUT_DEBUG("Points inside the outer ellipse and good gradient orientations");
      }
      // Create ellipse with its real size from original image.
      cctag::numerical::geometry::Ellipse rescaleEllipse(outerEllipse.center(), outerEllipse.a() * scale, outerEllipse.b() * scale, outerEllipse.angle());
      
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
              CCTAG_COUT_DEBUG( "Not enough outer ellipse points: realSizeOuterEllipsePoints : " << realSizeOuterEllipsePoints << ", rasterizeEllipsePerimeter : " << rasterizeEllipsePerimeter( outerEllipse )*scale << ", quality : " << quality );
              continue;
      }*/

      cctag::Point2dN<double> markerCenter;
      cctag::numerical::BoundedMatrix3x3d markerHomography;

      const double ratioSemiAxes = outerEllipse.a() / outerEllipse.b();

      if (ratioSemiAxes > 8.0 || ratioSemiAxes < 0.125)
      {
        CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RATIO_SEMIAXIS);
        CCTagFileDebug::instance().incrementFlowComponentIndex(0);
        CCTAG_COUT_DEBUG("Too high ratio between semi-axes!");
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

        CCTAG_COUT_DEBUG("Distance max to high!");
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

      CCTAG_COUT_DEBUG("------------------------------Added marker------------------------------");
    }
    catch (...)
    {
      CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(RAISED_EXCEPTION);
      CCTagFileDebug::instance().incrementFlowComponentIndex(0);
      // Ellipse fitting don't pass.
      CCTAG_COUT_CURRENT_EXCEPTION;
      CCTAG_COUT_DEBUG( "Exception raised" );
    }
  }

  boost::posix_time::ptime tstop2(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d2 = tstop2 - tstop1;
  const double spendTime2 = d2.total_milliseconds();
  CCTAG_COUT_OPTIM("Time in the 2nd loop" << spendTime2 << " ms");

  //	markers.sort();

  CCTAG_COUT_DEBUG("Markers creation time: " << t3.elapsed());
  POP_LEAVE;
}


void createImageForVoteResultDebug(
        const boost::gil::gray8_view_t & sourceView,
        const WinnerMap & winners)
{
#if defined(CCTAG_SERIALIZE)
  {

    POP_INFO << "running optional 'voting' block" << std::endl;
    std::size_t mx = 0;
    boost::gil::gray8_image_t vimg(sourceView.dimensions());
    boost::gil::gray8_view_t votevw(view(vimg));

    // Zero filling
    boost::gil::fill_black(votevw);

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
      *votevw.xy_at(winner->x(), winner->y()) = (unsigned char) ((v.size() * 10.0));
    }

    std::stringstream outFilenameVote;
    outFilenameVote << "voteLevel" << CCTagVisualDebug::instance().getPyramidLevel();
    CCTagVisualDebug::instance().initBackgroundImage(boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(votevw));
    CCTagVisualDebug::instance().newSession(outFilenameVote.str());

  }
#endif
}

void cctagDetection(CCTag::List& markers,
        const std::size_t frame, 
        const boost::gil::gray8_view_t& graySrc,
        const Parameters & params,
        const std::string & cctagBankFilename,
        const bool bDisplayEllipses)
{
  POP_ENTER;
  using namespace cctag;
  using namespace boost::numeric::ublas;
  using namespace boost::gil;
  
  std::srand(1);

  static const CCTagMarkersBank bank(cctagBankFilename);

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
#endif
  
#ifdef WITH_CUDA
  {
    popart::TagPipe pipe1;

    uint32_t w = graySrc.width();
    uint32_t h = graySrc.height();
    pipe1.prepframe( w, h, params );

    unsigned char* pix = new unsigned char[w*h];
    memset( pix, 0, w*h );

    boost::gil::copy_pixels( graySrc,
                             boost::gil::interleaved_view( graySrc.width(),
                                                           graySrc.height(),
                                                           pix,
                                                           graySrc.width() * sizeof(unsigned char) ) );

    pipe1.tagframe( pix, w, h, params );
    pipe1.debug( pix, params );
  }
#endif
  // Views for:
  // canny
  typedef kth_channel_view_type<0, rgb32f_view_t>::type CannyView;
  // x derivative
  typedef kth_channel_view_type<1, rgb32f_view_t>::type GradXView;
  // y derivative
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

  cannyCv(graySrc, cannyRGB, cannyView, cannyGradX, cannyGradY,
          params._cannyThrLow, params._cannyThrHigh);

  cctagMultiresDetection(markers, graySrc, cannyRGB, frame, params);

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());

  boost::posix_time::time_duration d = t1 - t0;
  const double spendTime = d.total_milliseconds();
  CCTAG_COUT_OPTIM("TIME IN DETECTION: " << spendTime << " ms");
#endif
  
  // Identification step
  // To decomment -- enable cuts selection, homography computation and identification
  if (params._doIdentification)
  {
    CCTag::List::iterator it = markers.begin();
    while (it != markers.end())
    {
      CCTag & cctag = *it;

      const int detected = cctag::identify(
              cctag,
              bank.getMarkers(),
              graySrc,
              cannyGradX,
              cannyGradY,
              params);
      
      cctag.setStatus(detected);

      try
      {
        std::vector<cctag::numerical::geometry::Ellipse> & ellipses = cctag.ellipses();

        bounded_matrix<double, 3, 3> mInvH;
        cctag::numerical::invert(cctag.homography(), mInvH);

        BOOST_FOREACH(double radiusRatio, cctag.radiusRatios())
        {
          cctag::numerical::geometry::Cercle circle(1.0 / radiusRatio);
          ellipses.push_back(cctag::numerical::geometry::Ellipse(
                  prec_prod(trans(mInvH), prec_prod<bounded_matrix<double, 3, 3> >(circle.matrix(), mInvH))));
        }

        // Push the outer ellipse
        ellipses.push_back(cctag.rescaledOuterEllipse());

        CCTAG_COUT_VAR_DEBUG(cctag.id());
        ++it;
      }
      catch (...)
      {
        // Impossible to construct Ellipse from computed homographies => conics are not ellipses!
        //it = markers.erase( it );
      }
    }
#ifdef CCTAG_OPTIM
      boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
      boost::posix_time::time_duration d2 = t2 - t1;
      const double spendTime2 = d2.total_milliseconds();
      CCTAG_COUT_OPTIM("TIME IN IDENTIFICATION: " << spendTime2 << " ms");
#endif
  }

  markers.sort();

  CCTagVisualDebug::instance().writeIdentificationView(markers);
  CCTagFileDebug::instance().newSession("identification.txt");

  BOOST_FOREACH(const CCTag & marker, markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }

  POP_LEAVE;
}

} // namespace cctag
