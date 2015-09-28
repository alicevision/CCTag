#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <limits>

#include <cctag/multiresolution.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/fileDebug.hpp>
#include <cctag/vote.hpp>
#include <cctag/ellipseGrowing.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>
#include <cctag/toolbox.hpp>
#include <cctag/image.hpp>
#include <cctag/canny.hpp>
#include <cctag/detection.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <sstream>

#ifdef WITH_CUDA
#ifdef WITH_CUDA_COMPARE_MODE
#include "cuda/tag.h"
#endif
#endif

namespace cctag
{

/* @brief Add markers from a list to another, deleting duplicates.
 *
 *
 */

bool intersectLineToTwoEllipses(
        std::ssize_t y,
        const numerical::geometry::Ellipse & qIn,
        const numerical::geometry::Ellipse & qOut,
        const EdgePointsImage & edgesMap,
        std::list<EdgePoint*> & pointsInHull)
{
  std::vector<double> intersectionsOut = numerical::geometry::intersectEllipseWithLine(qOut, y, true);
  std::vector<double> intersectionsIn = numerical::geometry::intersectEllipseWithLine(qIn, y, true);
  BOOST_ASSERT(intersectionsOut.size() <= 2);
  BOOST_ASSERT(intersectionsIn.size() <= 2);
  if ((intersectionsOut.size() == 2) && (intersectionsIn.size() == 2))
  {
    //@todo@Lilian, in/out the edgeMap
    std::ssize_t begin1 = std::max(0, (int) intersectionsOut[0]);
    std::ssize_t end1 = std::min((int) edgesMap.shape()[0] - 1, (int) intersectionsIn[0]);

    std::ssize_t begin2 = std::max(0, (int) intersectionsIn[1]);
    std::ssize_t end2 = std::min((int) edgesMap.shape()[0] - 1, (int) intersectionsOut[1]);

    for (int x = begin1; x <= end1; ++x)
    {
      EdgePoint* edgePoint = edgesMap[x][y];
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        if (boost::numeric::ublas::inner_prod(
                subrange(edgePoint->gradient(), 0, 2),
                subrange(qIn.center() - (*edgePoint), 0, 2)) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
    for (int x = begin2; x <= end2; ++x)
    {
      EdgePoint* edgePoint = edgesMap[x][y];
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        if (boost::numeric::ublas::inner_prod(
                subrange(edgePoint->gradient(), 0, 2),
                subrange(qIn.center() - (*edgePoint), 0, 2)) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else if ((intersectionsOut.size() == 2) && (intersectionsIn.size() <= 1))
  {
    std::ssize_t begin = std::max(0, (int) intersectionsOut[0]);
    std::ssize_t end = std::min((int) edgesMap.shape()[0] - 1, (int) intersectionsOut[1]);

    for (int x = begin; x <= end; ++x)
    {
      EdgePoint* edgePoint = edgesMap[x][y];
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        if (boost::numeric::ublas::inner_prod(
                subrange(edgePoint->gradient(), 0, 2),
                subrange(qIn.center() - (*edgePoint), 0, 2)) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else if ((intersectionsOut.size() == 1) && (intersectionsIn.size() == 0))
  {
    if ((intersectionsOut[0] >= 0) && (intersectionsOut[0] < edgesMap.shape()[0]))
    {
      EdgePoint* edgePoint = edgesMap[intersectionsOut[0]][y];
      if (edgePoint)
      {
        // Check that the gradient is opposed to the ellipse's center before pushing it.
        if (boost::numeric::ublas::inner_prod(
                subrange(edgePoint->gradient(), 0, 2),
                subrange(qIn.center() - (*edgePoint), 0, 2)) < 0)
        {
          pointsInHull.push_back(edgePoint);
        }
      }
    }
  }
  else //if( intersections.size() == 0 )
  {
    return false;
  }
  return true;
}

void selectEdgePointInEllipticHull(
        const EdgePointsImage & edgesMap,
        const numerical::geometry::Ellipse & outerEllipse,
        double scale,
        std::list<EdgePoint*> & pointsInHull)
{
  numerical::geometry::Ellipse qIn, qOut;
  computeHull(outerEllipse, scale, qIn, qOut);

  const double yCenter = outerEllipse.center().y();

  int maxY = std::max(int(yCenter), 0);
  int minY = std::min(int(yCenter), int(edgesMap.shape()[1]) - 1);

  // Visit the bottom part of the ellipse
  for (std::ssize_t y = maxY; y < int( edgesMap.shape()[1]); ++y)
  {
    if (!intersectLineToTwoEllipses(y, qIn, qOut, edgesMap, pointsInHull))
      break;
  }
  // Visit the upper part of the ellipse
  for (std::ssize_t y = minY; y >= 0; --y)
  {
    if (!intersectLineToTwoEllipses(y, qIn, qOut, edgesMap, pointsInHull))
      break;
  }
}

void update(
        CCTag::List& markers,
        const CCTag& markerToAdd)
{
  bool flag = false;

  BOOST_FOREACH(CCTag & currentMarker, markers)
  {
    // If markerToAdd is overlapping with a marker contained in markers then
    if (currentMarker.isOverlapping(markerToAdd))
    {
      if (markerToAdd.quality() > currentMarker.quality())
      {
        currentMarker = markerToAdd;
      }
      flag = true;
    }
  }
  // else push back in markers.
  if (!flag)
  {
    markers.push_back(new CCTag(markerToAdd));
  }
}

void cctagMultiresDetection_inner(
        size_t                  i,
        CCTag::List&            pyramidMarkers,
        const cv::Mat&          imgGraySrc,
        const Level*            level,
        const std::size_t       frame,
        std::vector<EdgePoint>& vPoints,
        EdgePointsImage&        vEdgeMap,
        popart::TagPipe*        cuda_pipe,
        const Parameters &      params )
{
    CCTAG_COUT_OPTIM(":::::::: Multiresolution level " << i << "::::::::");

    // Data structure for getting vote winners
    WinnerMap winners;
    std::vector<EdgePoint*> seeds;

#if defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    std::vector<EdgePoint>  cuda_debug_vPoints;
    EdgePointsImage         cuda_debug_vEdgeMap;
    WinnerMap               cuda_debug_winners;
    std::vector<EdgePoint*> cuda_debug_seeds;
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)

#if defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    // there is no point in measuring time in compare mode
    if( cuda_pipe ) {
        #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
        #endif
        cuda_pipe->download( i, 
                             cuda_debug_vPoints,
                             cuda_debug_vEdgeMap,
                             cuda_debug_seeds,
                             cuda_debug_winners );
        #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
        boost::posix_time::time_duration d = t1 - t0;
        const double spendTime = d.total_milliseconds();
        CCTAG_COUT_OPTIM("Time in GPU download: " << spendTime << " ms");
        #endif
    }
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)

#if defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    // there is no point in measuring time in compare mode
    if( cuda_pipe ) {
        #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
        #endif
        cuda_pipe->download( i, 
                             vPoints,
                             vEdgeMap,
                             seeds,
                             winners );
        #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
        boost::posix_time::time_duration d = t1 - t0;
        const double spendTime = d.total_milliseconds();
        CCTAG_COUT_OPTIM("Time in GPU download: " << spendTime << " ms");
        #endif
    }
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)

#if defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    if( not cuda_pipe ) {
#endif // defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    #ifdef CCTAG_OPTIM
    boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
    #endif
    edgesPointsFromCanny( vPoints,
                          vEdgeMap,
                          level->getEdges(),
                          level->getDx(),
                          level->getDy());
    #ifdef CCTAG_OPTIM
    boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration d = t1 - t0;
    const double spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in GPU Edge extraction: " << spendTime << " ms");
    #endif
#if defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    } // not cuda_pipe
#endif // defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)

    CCTagVisualDebug::instance().setPyramidLevel(i);

#if defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    if( cuda_pipe ) {
        std::cout << "Number of edge points: "
                  << vPoints.size() << " (CPU)"
                  << cuda_debug_vPoints.size() << " (GPU)" << std::endl;
        std::cout << "X-size: " << vEdgeMap.shape()[0]
                  << " Y-size: " << vEdgeMap.shape()[1] << " (CPU) "
                  << "X-size: " << cuda_debug_vEdgeMap.shape()[0]
                  << " Y-size: " << cuda_debug_vEdgeMap.shape()[1] << " (GPU)"
                  << std::endl;

        popart::TagPipe::debug_cpu_origin( i, level->getSrc(), params );
        popart::TagPipe::debug_cpu_edge_out( i, level->getEdges(), params );
        popart::TagPipe::debug_cpu_dxdy_out( cuda_pipe, i, level->getDx(), level->getDy(), params );
        popart::TagPipe::debug_cmp_edge_table( i, vEdgeMap, cuda_debug_vEdgeMap, params );
    }
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)

#if defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    if( not cuda_pipe ) {
#endif // defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    #ifdef CCTAG_OPTIM
    boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
    #endif
    // Voting procedure applied on every edge points.
    vote( vPoints,
          seeds,        // output
          vEdgeMap,
          winners,      // output
          level->getDx(),
          level->getDy(),
          params );
    
    if( seeds.size() > 1 ) {
        // Sort the seeds based on the number of received votes.
        std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);
    }
    #ifdef CCTAG_OPTIM
    boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration d = t1 - t0;
    const double spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in CPU vote and sort: " << spendTime << " ms");
    #endif
#if defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)
    } // not cuda_pipe
#endif // defined(WITH_CUDA) && not defined(WITH_CUDA_COMPARE_MODE)

#if defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    if( cuda_pipe ) {
        cctagDetectionFromEdges(
                pyramidMarkers,
                cuda_debug_vPoints,
                level->getSrc(),
                cuda_debug_winners,
                cuda_debug_seeds,
                cuda_debug_vEdgeMap,
                frame, i, std::pow(2.0, (int) i), params);
    } else {
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    cctagDetectionFromEdges(
        pyramidMarkers,
        vPoints,
        level->getSrc(),
        winners,
        seeds,
        vEdgeMap,
        frame, i, std::pow(2.0, (int) i), params);
#if defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    }
#endif // defined(WITH_CUDA) && defined(WITH_CUDA_COMPARE_MODE)
    
    CCTagVisualDebug::instance().initBackgroundImage(level->getSrc());
    std::stringstream outFilename2;
    outFilename2 << "viewLevel" << i;
    CCTagVisualDebug::instance().newSession(outFilename2.str());

    BOOST_FOREACH(const CCTag & marker, pyramidMarkers)
    {
        CCTagVisualDebug::instance().drawMarker(marker, false);
    }
}

void cctagMultiresDetection(
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid& imagePyramid,
        const std::size_t   frame,
        popart::TagPipe*    cuda_pipe,
        const Parameters&   params)
{
    // POP_ENTER;
  //	* For each pyramid level:
  //	** launch CCTag detection based on the canny edge detection output.
  
  bool doUpdate = true; // todo@Lilian: add in the parameter file.

  std::map<std::size_t, CCTag::List> pyramidMarkers;
  std::vector<EdgePointsImage> vEdgeMaps;
  vEdgeMaps.reserve(imagePyramid.getNbLevels());
  std::vector<std::vector<EdgePoint > > vPoints;

  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );
  for ( std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers; ++i ) {
    pyramidMarkers.insert( std::pair<std::size_t, CCTag::List>( i, CCTag::List() ) );

    // Create EdgePoints for every detected edge points in edges.
    // std::vector<EdgePoint> points;
    // vPoints.push_back(points);
    vPoints.push_back( std::vector<EdgePoint>() );
    
    // EdgePointsImage edgesMap;
    // vEdgeMaps.push_back(edgesMap);
    vEdgeMaps.push_back( EdgePointsImage() );
    
    cctagMultiresDetection_inner( i,
                                  pyramidMarkers[i],
                                  imgGraySrc,
                                  imagePyramid.getLevel(i),
                                  frame,
                                  vPoints.back(),
                                  vEdgeMaps.back(),
                                  cuda_pipe,
                                  params );
  }
  
  // Delete overlapping markers while keeping the best ones.
  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );
  for (std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers ; ++i)
  // set the _numberOfProcessedMultiresLayers <= _numberOfMultiresLayers todo@Lilian
  {
    CCTag::List & markersList = pyramidMarkers[i];

    BOOST_FOREACH(const CCTag & marker, markersList)
    {
      if (doUpdate)
      {
        update(markers, marker);
      }
      else
      {
        markers.push_back(new CCTag(marker));
      }
    }
  }
  
  CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(0)->getSrc());
  CCTagVisualDebug::instance().writeLocalizationView(markers);

  // Final step: extraction of the detected markers in the original (scale) image.
  CCTagVisualDebug::instance().newSession("multiresolution");

  // Project markers from the top of the pyramid to the bottom (original image).
  BOOST_FOREACH(CCTag & marker, markers)
  {
    int i = marker.pyramidLevel();
    // if the marker has to be rescaled into the original image
    if (i > 0)
    {
      BOOST_ASSERT( i < params._numberOfMultiresLayers );
      double scale = marker.scale(); // pow( 2.0, (double)i );

      cctag::numerical::geometry::Ellipse rescaledOuterEllipse = marker.rescaledOuterEllipse();

      std::list<EdgePoint*> pointsInHull;
      selectEdgePointInEllipticHull(vEdgeMaps[0], rescaledOuterEllipse, scale, pointsInHull);

      std::vector<EdgePoint*> rescaledOuterEllipsePoints;

      double SmFinal = 1e+10;
      
      cctag::outlierRemoval(pointsInHull, rescaledOuterEllipsePoints, SmFinal, 20.0);
      
      try
      {
        numerical::ellipseFitting(rescaledOuterEllipse, rescaledOuterEllipsePoints);

        std::vector< Point2dN<double> > rescaledOuterEllipsePointsDouble;// todo@Lilian : add a reserve
        std::size_t numCircles = params._nCrowns * 2;

        BOOST_FOREACH(EdgePoint * e, rescaledOuterEllipsePoints)
        {
          rescaledOuterEllipsePointsDouble.push_back(Point2dN<double>(e->x(), e->y()));
          CCTagVisualDebug::instance().drawPoint(Point2dN<double>(e->x(), e->y()), cctag::color_red);
        }
        marker.setCenterImg(cctag::Point2dN<double>(marker.centerImg().getX() * scale, marker.centerImg().getY() * scale));
        marker.setRescaledOuterEllipse(rescaledOuterEllipse);
        marker.setRescaledOuterEllipsePoints(rescaledOuterEllipsePointsDouble);
      }
      catch (...)
      {
        // catch exception from ellipseFitting
      }
    }
    else
    {
      marker.setRescaledOuterEllipsePoints(marker.points().back());
    }
  }
  
  // Log
  CCTagFileDebug::instance().newSession("data.txt");
  BOOST_FOREACH(const CCTag & marker, markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }
  
  // POP_LEAVE;
  
}

void clearDetectedMarkers(
        const std::map<std::size_t,
        CCTag::List> & pyramidMarkers,
        const boost::gil::rgb32f_view_t & cannyRGB,
        const std::size_t curLevel )
{
  using namespace boost::gil;
  typedef rgb32f_pixel_t Pixel;
  Pixel pixelZero;
  terry::numeric::pixel_zeros_t<Pixel>()( pixelZero );
  typedef std::map<std::size_t, CCTag::List> LeveledMarkersT;

  BOOST_FOREACH( const LeveledMarkersT::const_iterator::value_type & v, pyramidMarkers )
  {
    const std::size_t level = v.first;
    const double factor = std::pow( 2.0, (double)(curLevel - level) );
    const CCTag::List & markers = v.second;
    BOOST_FOREACH( const CCTag & tag, markers )
    {
      BOOST_FOREACH( const cctag::numerical::geometry::Ellipse & ellipse, tag.ellipses() )
      {
        cctag::numerical::geometry::Ellipse ellipseScaled = ellipse;
        // Scale center
        Point2dN<double> c = ellipseScaled.center();
        c.setX( c.x() * factor );
        c.setY( c.y() * factor );
        ellipseScaled.setCenter( c );
        // Scale demi axes
        ellipseScaled.setA( ellipseScaled.a() * factor );
        ellipseScaled.setB( ellipseScaled.b() * factor );
        // Erase ellipses
        fillEllipse( cannyRGB, ellipseScaled, pixelZero );
      }
    }
  }
}

} // namespace cctag

