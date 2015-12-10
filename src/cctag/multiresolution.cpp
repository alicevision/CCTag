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
#include <cctag/talk.hpp> // for DO_TALK macro
#include <cctag/package.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <sstream>
#include <fstream>
#include <map>

#ifdef WITH_CUDA
#include <cuda_runtime.h> // only for debugging!!!
#include "cuda/tag.h"
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

void cctagMultiresDetection_download(
        popart::Package*        package,
        size_t                  i,
        Level*                  level,
        std::vector<EdgePoint>& vPoints,
        EdgePointsImage&        vEdgeMap,
        popart::TagPipe*        cuda_pipe,
        const Parameters &      params,
        cctag::logtime::Mgmt*   durations )
{
    // WinnerMap winners;
    // std::vector<EdgePoint*> seeds;
#if defined(WITH_CUDA)
    if( cuda_pipe ) {
        cuda_pipe->convertToHost( i, 
                                  vPoints,
                                  vEdgeMap,
                                  package->seeds[i],
                                  package->winners[i] );
        if( durations ) {
            cudaDeviceSynchronize();
        }
        level->setLevel( cuda_pipe, params );

        CCTagVisualDebug::instance().setPyramidLevel(i);
    }
#endif // defined(WITH_CUDA)
}

void cctagMultiresDetection_inner(
        popart::Package*        package,
        size_t                  i,
        CCTag::List&            pyramidMarkers,
        const cv::Mat&          imgGraySrc,
        Level*                  level,
        const std::size_t       frame,
        std::vector<EdgePoint>& vPoints,
        EdgePointsImage&        vEdgeMap,
        bool                    cuda_computations,
        const Parameters &      params,
        cctag::logtime::Mgmt*   durations )
{
    DO_TALK( CCTAG_COUT_OPTIM(":::::::: Multiresolution level " << i << "::::::::"); )

    // Data structure for getting vote winners
    WinnerMap&               winners( package->winners[i] );
    std::vector<EdgePoint*>& seeds( package->seeds[i] );

    boost::posix_time::time_duration d;

#if defined(WITH_CUDA)
    // there is no point in measuring time in compare mode
    if( not cuda_computations ) {
#endif // defined(WITH_CUDA)
    edgesPointsFromCanny( vPoints,
                          vEdgeMap,
                          level->getEdges(),
                          level->getDx(),
                          level->getDy());

    CCTagVisualDebug::instance().setPyramidLevel(i);

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

#if defined(WITH_CUDA)
    } // not cuda_pipe
#endif // defined(WITH_CUDA)


    cctagDetectionFromEdges(
        pyramidMarkers,
        vPoints,
        level->getSrc(),
        winners,
        seeds,
        vEdgeMap,
        frame, i, std::pow(2.0, (int) i), params,
        durations );

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
        popart::Package* package,
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid& imagePyramid,
        const std::size_t   frame,
        popart::TagPipe*    cuda_pipe,
        const Parameters&   params,
        cctag::logtime::Mgmt* durations )
{
  //	* For each pyramid level:
  //	** launch CCTag detection based on the canny edge detection output.
  
  bool doUpdate = true; // todo@Lilian: add in the parameter file.

  // std::map<std::size_t, CCTag::List> pyramidMarkers;
  // std::vector<EdgePointsImage> vEdgeMaps;
  // vEdgeMaps.reserve(imagePyramid.getNbLevels());
  // std::vector<std::vector<EdgePoint > > vPoints;
  CCTag::List            pyramidMarkers[ params._numberOfProcessedMultiresLayers ];
  EdgePointsImage        vEdgeMaps     [ params._numberOfProcessedMultiresLayers ];
  std::vector<EdgePoint> vPoints       [ params._numberOfProcessedMultiresLayers ];

  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );

  for ( std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers; ++i ) {
    // pyramidMarkers.insert( std::pair<std::size_t, CCTag::List>( i, CCTag::List() ) );

    // Create EdgePoints for every detected edge points in edges.
    // std::vector<EdgePoint> points;
    // vPoints.push_back(points);
    // vPoints.push_back( std::vector<EdgePoint>() );
    
    // EdgePointsImage edgesMap;
    // vEdgeMaps.push_back(edgesMap);
    // vEdgeMaps.push_back( EdgePointsImage() );
    
    cctagMultiresDetection_download( package,
                                     i,
                                     imagePyramid.getLevel(i),
                                     vPoints[i],
                                     vEdgeMaps[i],
                                     cuda_pipe,
                                     params,
                                     durations );
  }
  package->unlockPhase1( );

  for ( std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers; ++i ) {
    cctagMultiresDetection_inner( package,
                                  i,
                                  pyramidMarkers[i],
                                  imgGraySrc,
                                  imagePyramid.getLevel(i),
                                  frame,
                                  vPoints[i], // vPoints.back(),
                                  vEdgeMaps[i], // vEdgeMaps.back(),
                                  ( cuda_pipe != 0 ),
                                  params,
                                  durations );
  }
  if( durations ) durations->log( "after cctagMultiresDetection_inner" );
  
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
  if( durations ) durations->log( "after update markers" );
  
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

      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
      #endif
      
      
      std::list<EdgePoint*> pointsInHull;
      selectEdgePointInEllipticHull(vEdgeMaps[0], rescaledOuterEllipse, scale, pointsInHull);

      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
      #endif
      
      std::vector<EdgePoint*> rescaledOuterEllipsePoints;

      double SmFinal = 1e+10;
      
      cctag::outlierRemoval(
              pointsInHull,
              rescaledOuterEllipsePoints,
              SmFinal,
              20.0,
              NO_WEIGHT,
              60); 
      
      #ifdef CCTAG_OPTIM
        boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
        boost::posix_time::time_duration d1 = t1 - t0;
        boost::posix_time::time_duration d2 = t2 - t1;
        CCTAG_COUT_OPTIM("Time in selectEdgePointInEllipticHull: " << d1.total_milliseconds() << " ms");
        CCTAG_COUT_OPTIM("Time in outlierRemoval: " << d2.total_milliseconds() << " ms");
      #endif
      
      try
      {
        numerical::ellipseFitting(rescaledOuterEllipse, rescaledOuterEllipsePoints);

        std::vector< DirectedPoint2d<double> > rescaledOuterEllipsePointsDouble;// todo@Lilian : add a reserve
        std::size_t numCircles = params._nCrowns * 2;

        BOOST_FOREACH(EdgePoint * e, rescaledOuterEllipsePoints)
        {
          rescaledOuterEllipsePointsDouble.push_back(
                  DirectedPoint2d<double>(e->x(), e->y(),
                  e->_grad.x(),
                  e->_grad.y())
          );
          
          CCTagVisualDebug::instance().drawPoint(Point2dN<double>(e->x(), e->y()), cctag::color_red);
        }
        //marker.setCenterImg(rescaledOuterEllipse.center());                                                                // todo
        marker.setCenterImg(cctag::Point2dN<double>(marker.centerImg().getX() * scale, marker.centerImg().getY() * scale));  // decide between these two lines
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
  if( durations ) durations->log( "after marker projection" );
  
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

