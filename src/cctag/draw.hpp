#ifndef VISION_CCTAG_DRAW_HPP_
#define VISION_CCTAG_DRAW_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/boostCv/cvImage.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>

#include <boost/gil/image.hpp>
#include <boost/foreach.hpp>

#include <vector>


namespace cctag {

template<class View>
static bool fillEllipseLine( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
{
  std::vector<double> intersections = numerical::geometry::intersectEllipseWithLine( ellipse, y, true );
  BOOST_ASSERT( intersections.size() <= 2 );
  if( intersections.size() == 2 )
  {
          std::fill( image.at(intersections[0], y), image.at(intersections[1], y), color );
  }
  else if( intersections.size() == 1 )
  {
          *image.at(intersections[0], y) = color;
  }
  else //if( intersections.size() == 0 )
  {
          return false;
  }
  return true;
}


template<class View>
static bool drawEllipseLineIntersections( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
{
  std::vector<double> intersections = numerical::geometry::intersectEllipseWithLine( ellipse, y, true );
  BOOST_ASSERT( intersections.size() <= 2 );
  if( intersections.size() == 2 )
  {
          if( intersections[0] >= 0 && intersections[0] < image.width() )
                  *image.at(intersections[0], y) = color;
          if( intersections[1] >= 0 && intersections[1] < image.width() )
                  *image.at(intersections[1], y) = color;
  }
  else if( intersections.size() == 1 )
  {
          if( intersections[0] >= 0 && intersections[0] < image.width() )
                  *image.at(intersections[0], y) = color;
  }
  else //if( intersections.size() == 0 )
  {
          return false;
  }
  return true;
}

/**
 * @brief Fill image under an ellipse with \param color.
 *
 * @param[out] image output image we fill in \param color.
 * @param[in] ellipse define the region to fill.
 * @param[in] color color used to fill.
 */
template<class View>
inline void fillEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
{
  const double yCenter = ellipse.center().y();

  // visit the bottom part of the ellipse
  for( std::ssize_t y = yCenter; y < image.height(); ++y )
  {
          if( ! fillEllipseLine( image, ellipse, color, y ) )
                  break;
  }
  // visit the upper part of the ellipse
  for( std::ssize_t y = yCenter; y > 0; --y )
  {
          if( ! fillEllipseLine( image, ellipse, color, y ) )
                  break;
  }
}

template<class View>
inline void drawEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
{
  std::vector< Point2dN<int> > vPoint;
  numerical::geometry::rasterizeEllipse( ellipse, vPoint );

  BOOST_FOREACH(Point2dN<int> p, vPoint){
    if( (p.x() >= 0) && (p.x() < image.width()) && (p.y() >= 0) && (p.y() < image.height()) ){
            *image.at(p.x(), p.y()) = color;
    }
  }
}

} // namespace cctag

#endif
