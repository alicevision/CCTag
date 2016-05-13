#ifndef _CCTAG_NUMERICAL_ELLIPSEFROMPOINTS_HPP_
#define _CCTAG_NUMERICAL_ELLIPSEFROMPOINTS_HPP_

#include <cctag/geometry/Ellipse.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>

#include <cstddef>
#include <cmath>
#include <vector>


namespace cctag {
namespace numerical {
namespace geometry {

Point2d<Eigen::Vector3f> extractEllipsePointAtAngle( const Ellipse & ellipse, float theta );
///@todo rename this function
void points( const Ellipse & ellipse, const std::size_t nb, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );
///@todo rename this function
void points( const Ellipse & ellipse, const std::size_t nb, const float phi1, const float phi2, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );

template <class T>
void fitEllipse(const std::vector<cctag::Point2d<T>> & points, Ellipse& e )
{
	std::vector<cv::Point2f> cvpts;
	cvpts.reserve( points.size() );
        for (const auto& pt : points)
		cvpts.push_back( cv::Point2f(pt.x(), pt.y()));
        
	const cv::RotatedRect rR = cv::fitEllipse( cv::Mat( cvpts ) );
	const float xC           = rR.center.x;
	const float yC           = rR.center.y;

	const float b = rR.size.height / 2.0;
	const float a = rR.size.width / 2.0;

	const float angle = rR.angle * boost::math::constants::pi<float>() / 180.0;

	e.setParameters( Point2d<Eigen::Vector3f>( xC, yC ), a, b, angle );
}

// Direct implementation of "Direct Least Squares Fitting of Ellipses" by Fitzgibbon
// Was previously THE method used in opencv; see commit
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
// D is the "design matrix" with points offset by center.
bool fitEllipse(Eigen::Matrix<float, Eigen::Dynamic, 6>& D, const Eigen::Vector2f center, Ellipse& ellipse);

// Converts the input to the design matrix required by the above overload and calls the overload.
template<typename EV> // EV is derived from Eigen::Matrix<T, 3, 1>
bool fitEllipse(const EV* points, const size_t n, Ellipse& e)
{
  using Array6 = Eigen::Array<float, 6, 1>;
  using Vector6 = Eigen::Matrix<float, 6, 1>;
  
  // Find the center.
  Eigen::Vector2f center(0, 0);
  for (size_t i = 0; i < n; ++i)
    center += Eigen::Vector2f(points[i](0), points[i](1));
  center /= n;
  
  // D: The design matrix; rows are (x*x, x*y, y*y, x, y, 1); x/y shifted by center
  Eigen::Matrix<float, Eigen::Dynamic, 6> D(n, 6);
  {
    Array6 d1, d2, offset1, offset2;
    offset1 << center(0), center(0), center(1), center(0), center(1), 0;
    offset2 << center(0), center(1), center(1), center(0), center(1), 0;
    
    for (size_t i = 0; i < n; ++i) {
      const auto& p = points[i];
      d1 << p(0), p(0), p(1), p(0), p(1), 1.f;
      d2 << p(0), p(1), p(1), p(0), p(1), 1.f;
      D.row(i) = ((d1-offset1) * (d2-offset2)).matrix();
    }
  }

  return fitEllipse(D, center, e);
}

void ellipsePoint( const cctag::numerical::geometry::Ellipse& ellipse, float theta, Eigen::Vector3f& pt );
void computeIntermediatePoints(const Ellipse & ellipse, Point2d<Eigen::Vector3i> & pt11, Point2d<Eigen::Vector3i> & pt12, Point2d<Eigen::Vector3i> & pt21, Point2d<Eigen::Vector3i> & pt22);
void rasterizeEllipticalArc(const Ellipse & ellipse, const Point2d<Eigen::Vector3i> & pt1, const Point2d<Eigen::Vector3i> & pt2, std::vector< Point2d<Eigen::Vector3i> > & vPoint, std::size_t intersectionIndex);
void rasterizeEllipse( const Ellipse & ellipse, std::vector< Point2d<Eigen::Vector3i> > & vPoint );

/**
 * Get the pixel perimeter of an ellipse
 * @param ellipse
 * @return the perimeter in number of pixels
 */
std::size_t rasterizeEllipsePerimeter( const Ellipse & ellipse );

/**
 * @brief Compute intersections if there, between a line of equation Y = y and an ellipse.
 *
 * @param[in] ellipse
 * @param[in] y ordonate for which we compute x values of intersections
 * @return intersected points sorted in ascend order (returns x coordinates: 0, 1, or 2 points).
 */
std::vector<float> intersectEllipseWithLine( const numerical::geometry::Ellipse& ellipse, const float y, bool horizontal);

}
}
}

#endif

