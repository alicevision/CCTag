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
        const boost::gil::gray8_view_t& graySrc,
        const boost::gil::rgb32f_view_t & cannyRGB,
        const std::size_t frame,
        const Parameters & params)
{
  POP_ENTER;
  bool doUpdate = true;

  using namespace boost::gil;

  typedef kth_channel_view_type<0, rgb32f_view_t>::type CannyView;
  typedef kth_channel_view_type<1, rgb32f_view_t>::type GradXView;
  typedef kth_channel_view_type<2, rgb32f_view_t>::type GradYView;

  //	* create all pyramid levels
  //gray8_image_t grayImg;
  //gray8_view_t graySrc = img::toGray(grayImg, grayImg);

  PyramidImage<gray8_view_t> multiresSrc(
          graySrc,
          params._numberOfMultiresLayers);
  
  PyramidImage<rgb32f_view_t> multiresCanny(
          graySrc.width(),
          graySrc.height(),
          params._numberOfMultiresLayers);

  // todo@Lilian: add more description
  //	* for each pyramid level (except full image)
  //	** launch CCTag detection (from canny)
  //	** fill canny with ellipses in black
  //
  //	* for the full image
  //	** scale all previously detected ellipses
  //	** compute ellipse hull
  //	** collect outer edge points inside ellipse hull (ellipseHull)
  //	*** intersecter chaque ligne et parcours de ce fragment
  //	** erase canny under ellipse
  //	*** we can reuse previous intersections
  //	** launch CCTag detection (from canny)

  typedef rgb32f_pixel_t Pixel;

  Pixel pixelZero;
  terry::numeric::pixel_zeros_t<Pixel>()(pixelZero);
  std::map<std::size_t, CCTag::List> pyramidMarkers;

  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );
  for ( std::size_t i = params._numberOfMultiresLayers;
          i > std::max<size_t>(params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers, 1);
            --i)
  {

    CCTAG_COUT_OPTIM(":::::::::::::::: Multiresolution level " << i - 1 <<
             " :::::::::::::::::::");

    CCTag::List & markersList = pyramidMarkers[i - 1];

    rgb32f_view_t subCannyRGB = multiresCanny.getView(i - 1);

    CannyView cannyView;
    GradXView cannyGradX;
    GradYView cannyGradY;

    cannyView = kth_channel_view<0>(subCannyRGB);
    // x gradient
    cannyGradX = kth_channel_view<1>(subCannyRGB);
    // y gradient
    cannyGradY = kth_channel_view<2>(subCannyRGB);

    cannyCv(multiresSrc.getView(i - 1), subCannyRGB, cannyView,
            cannyGradX, cannyGradY, params._cannyThrLow, params._cannyThrHigh);

#ifdef CCTAG_SERIALIZE
    std::stringstream outFilenameCanny;
    outFilenameCanny << "cannyLevel" << i - 1;
    
    CCTagVisualDebug::instance().initBackgroundImage(
            boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(cannyView));
    
    CCTagVisualDebug::instance().newSession(outFilenameCanny.str());

#endif

    std::vector<EdgePoint> points;
    EdgePointsImage edgesMap;
    edgesPointsFromCanny(points, edgesMap, cannyView, cannyGradX, cannyGradY);

    CCTagVisualDebug::instance().setPyramidLevel(i - 1);

    cctagDetectionFromEdges(markersList, points,
            multiresSrc.getView(i - 1), cannyGradX, cannyGradY, edgesMap,
            frame, i - 1, std::pow(2.0, (int) i - 1), params);

    CCTAG_COUT("cctagDetection on layer 1:");
    CCTAG_COUT_VAR(markersList.size());

    CCTagVisualDebug::instance().initBackgroundImage(multiresSrc.getView(i - 1));
    std::stringstream outFilename2;
    outFilename2 << "viewLevel" << i - 1;
    CCTagVisualDebug::instance().newSession(outFilename2.str());

    //update(markers, markersList);

    BOOST_FOREACH(const CCTag & marker, markersList)
    {
      CCTagVisualDebug::instance().drawMarker(marker, false);
      //update(markers, marker);
    }
  }

  CCTAG_COUT_OPTIM(":::::::::::::::: Multiresolution level 0 :::::::::::::::::::");

  rgb32f_view_t subCannyRGB = multiresCanny.getView(0);
  CannyView cannyView;
  GradXView cannyGradX;
  GradYView cannyGradY;

  cannyView = kth_channel_view<0>(cannyRGB);
  // x gradient
  cannyGradX = kth_channel_view<1>(cannyRGB);
  // y gradient
  cannyGradY = kth_channel_view<2>(cannyRGB);

  cannyCv(multiresSrc.getView(0), subCannyRGB, cannyView, cannyGradX,
          cannyGradY, params._cannyThrLow, params._cannyThrHigh);

#ifdef CCTAG_SERIALIZE
  CCTagVisualDebug::instance().initBackgroundImage(
          boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(cannyView));
  CCTagVisualDebug::instance().newSession("cannyLevel0");
#endif

  std::vector<EdgePoint> points;
  EdgePointsImage edgesMap;
  edgesPointsFromCanny(points, edgesMap, cannyView, cannyGradX, cannyGradY);

  if (params._numberOfMultiresLayers == params._numberOfProcessedMultiresLayers)
  {

    CCTag::List & markersList = pyramidMarkers[0];

    CCTagVisualDebug::instance().setPyramidLevel(0);

    cctagDetectionFromEdges(markersList, points, multiresSrc.getView(0),
            cannyGradX, cannyGradY, edgesMap, frame, 0, 1.0, params);
    CCTAG_COUT_OPTIM("After 2st cctagDetection");
    CCTAG_COUT_VAR_OPTIM(markersList.size());

    CCTagVisualDebug::instance().initBackgroundImage(multiresSrc.getView(0));
    CCTagVisualDebug::instance().newSession("viewLevel0");

    BOOST_FOREACH(const CCTag & marker, markersList)
    {
      CCTagVisualDebug::instance().drawMarker(marker, false);

      if (doUpdate)
      {
        CCTAG_COUT_DEBUG("Update Level 0");
        update(markers, marker);
      }
      else
      {
        markers.push_back(new CCTag(marker));
      }
    }
  }

  BOOST_ASSERT( params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers >= 0 );
  for (std::size_t i = params._numberOfMultiresLayers - 1 ;
          i >= std::max<size_t>(params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers, 1);
            --i)
  {
    CCTag::List & markersList = pyramidMarkers[i];

    BOOST_FOREACH(const CCTag & marker, markersList)
    {
      if (doUpdate)
      {
        CCTAG_COUT_DEBUG("Update Level " << i);
        update(markers, marker);
      }
      else
      {
        markers.push_back(new CCTag(marker));
      }
    }
  }

  CCTagVisualDebug::instance().writeLocalizationView(markers);

  // Final step: extraction of the detected markers in the original (scale) image.

  CCTagVisualDebug::instance().newSession("multiresolution");

  // Project markers from the top of the pyramid to the bottom (original image).
  //for (std::size_t i = params._numberOfMultiresLayers - 1; i >= max(params._numberOfMultiresLayers - params._numberOfProcessedMultiresLayers, 1); --i) {

  //CCTag::List & markersList = pyramidMarkers[i];

  BOOST_FOREACH(CCTag & marker, markers)
  {

    int i = marker.pyramidLevel();

    // if the marker has to be rescaled into the original image
    if ((i > 0) && (i < params._numberOfMultiresLayers))
    {

      double scale = marker.scale(); //std::pow( 2.0, (double)i );

      cctag::numerical::geometry::Ellipse rescaledOuterEllipse = marker.rescaledOuterEllipse();

      std::list<EdgePoint*> pointsInHull;
      selectEdgePointInEllipticHull(edgesMap, rescaledOuterEllipse, scale, pointsInHull);

      std::vector<EdgePoint*> rescaledOuterEllipsePoints;

      double SmFinal = 1e+10;

      cctag::outlierRemoval(pointsInHull, rescaledOuterEllipsePoints, SmFinal, 20.0);

      // Optional
      //std::vector<EdgePoint*> outerEllipsePointsGrowing;
      //{
      //cctag::ellipseGrowing( edgesMap, outerEllipsePoints, outerEllipsePointsGrowing, outerEllipse, scale );
      //outerEllipsePoints.clear();
      //cctag::outlierRemoval( outerEllipsePointsGrowing, outerEllipsePoints, 20.0 ); // PB, move list to vector in this function or inverse in ellipseGrowing @Lilian
      //}


      try
      {
        numerical::ellipseFitting(rescaledOuterEllipse, rescaledOuterEllipsePoints);

        std::vector< Point2dN<double> > rescaledOuterEllipsePointsDouble;// todo@Lilian : add a reserve
        std::size_t numCircles = params._numCrowns * 2;
        //pointsCCTag.resize(numCircles);

        BOOST_FOREACH(EdgePoint * e, rescaledOuterEllipsePoints)
        {

          rescaledOuterEllipsePointsDouble.push_back(Point2dN<double>(e->x(), e->y()));
          //pointsCCTag[numCircles - 1].push_back(Point2dN<double>(e->x(), e->y()));

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

  //CCTagVisualDebug::instance().newSession("FinalMultiRes");

  CCTagFileDebug::instance().newSession("data.txt");

  BOOST_FOREACH(const CCTag & marker, markers)
  {
    CCTagFileDebug::instance().outputMarkerInfos(marker);
  }

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

