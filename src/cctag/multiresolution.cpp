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

void cctagMultiresDetection(
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid & imagePyramid,
        const std::size_t frame,
        const Parameters & params)
{
    POP_ENTER;
  //	* For each pyramid level:
  //	** launch CCTag detection based on the canny edge detection output.
  
  bool doUpdate = true; // todo@Lilian: add in the parameter file.

  std::map<std::size_t, CCTag::List> pyramidMarkers;
  std::vector<EdgePointsImage> vEdgeMaps;
  vEdgeMaps.reserve(imagePyramid.getNbLevels());
  
  for ( std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers; ++i ) //todo@Lilian : to move
  {
    EdgePointsImage edgesMap;
    edgesMap.resize( boost::extents[imagePyramid.getLevel(i)->width()][imagePyramid.getLevel(i)->height()] );
    std::fill( edgesMap.origin(), edgesMap.origin() + edgesMap.size(), (EdgePoint*)NULL );
    vEdgeMaps.push_back(edgesMap);
  }

  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );
  for ( std::size_t i = 0 ; i < params._numberOfProcessedMultiresLayers; ++i )
  {
    CCTAG_COUT_OPTIM(":::::::: Multiresolution level " << i << "::::::::");

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
#endif
    
    // Create EdgePoints for every detected edge points in edges.
    std::vector<EdgePoint> points;
    
    edgesPointsFromCanny( points, vEdgeMaps[i],
            imagePyramid.getLevel(i)->getEdges(),
            imagePyramid.getLevel(i)->getDx(),
            imagePyramid.getLevel(i)->getDy());

    CCTagVisualDebug::instance().setPyramidLevel(i);
    
    //cctagDetectionFromEdges(markersList, points,
    //        multiresSrc.getView(i - 1), cannyGradX, cannyGradY, edgesMap,
    //        frame, i - 1, std::pow(2.0, (int) i - 1), params);
    
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d = t1 - t0;
  const double spendTime = d.total_milliseconds();
  CCTAG_COUT_OPTIM("Time in edge point collection: " << spendTime << " ms");
#endif
    
    cctagDetectionFromEdges(
            pyramidMarkers[i], points,
            imagePyramid.getLevel(i)->getSrc(),
            imagePyramid.getLevel(i)->getDx(),
            imagePyramid.getLevel(i)->getDy(),
            vEdgeMaps[i],
            frame, i, std::pow(2.0, (int) i), params);
    
    CCTAG_COUT_VAR(pyramidMarkers[i].size());

    CCTAG_COUT(" ---------------------------------- 1 --------");
    
    CCTagVisualDebug::instance().initBackgroundImage(imagePyramid.getLevel(i)->getSrc());
    std::stringstream outFilename2;
    outFilename2 << "viewLevel" << i;
    CCTagVisualDebug::instance().newSession(outFilename2.str());

    BOOST_FOREACH(const CCTag & marker, pyramidMarkers[i])
    {
      CCTagVisualDebug::instance().drawMarker(marker, false);
    }
  }

  CCTAG_COUT("---------------------------------- 2 --------");
  
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

  CCTAG_COUT("---------------------------------- 4 --------");

  // Project markers from the top of the pyramid to the bottom (original image).
  BOOST_FOREACH(CCTag & marker, markers)
  {
    int i = marker.pyramidLevel();
    // if the marker has to be rescaled into the original image
    if (i > 0)
    {
      CCTAG_COUT("---------------------------------- 41 --------");
      BOOST_ASSERT( i < params._numberOfMultiresLayers );
      CCTAG_COUT("---------------------------------- 42 --------");
      double scale = marker.scale(); // pow( 2.0, (double)i );

      cctag::numerical::geometry::Ellipse rescaledOuterEllipse = marker.rescaledOuterEllipse();

      std::list<EdgePoint*> pointsInHull;
      CCTAG_COUT(" ---------------------------------- 5-1 --------");
      selectEdgePointInEllipticHull(vEdgeMaps[0], rescaledOuterEllipse, scale, pointsInHull);
      CCTAG_COUT("---------------------------------- 5-2 --------");

      std::vector<EdgePoint*> rescaledOuterEllipsePoints;

      double SmFinal = 1e+10;
      
      cctag::outlierRemoval(pointsInHull, rescaledOuterEllipsePoints, SmFinal, 20.0);
      
      CCTAG_COUT("---------------------------------- 6 --------");
      
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

  CCTAG_COUT("---------------------------------- 7 --------");
  
  // Log
  CCTagFileDebug::instance().newSession("data.txt");
  BOOST_FOREACH(const CCTag & marker, markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }

  CCTAG_COUT("---------------------------------- 8 --------");
  
  POP_LEAVE;
  
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

