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
#include <Eigen/Eigenvalues>


namespace cctag {
namespace numerical {
namespace geometry {

Point2d<Eigen::Vector3f> extractEllipsePointAtAngle( const Ellipse & ellipse, float theta );
///@todo rename this function
void points( const Ellipse & ellipse, const std::size_t nb, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );
///@todo rename this function
void points( const Ellipse & ellipse, const std::size_t nb, const float phi1, const float phi2, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts );


// Direct implementation of "Direct Least Squares Fitting of Ellipses" by Fitzgibbon
// Was previously THE method used in opencv; see commit
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
template<typename EV> // EV is derived from Eigen::Matrix<T, 3, 1>
void fitEllipse(const EV* points, const size_t n, Ellipse& e)
{
  using Array6 = Eigen::Array<float, 6, 1>;
  using Vector6 = Eigen::Matrix<float, 6, 1>;
  using Matrix6 = Eigen::Matrix<float, 6, 6>;
  
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
  
  // S: The scatter matrix.
  Matrix6 S = D.transpose() * D;
  
  // C: The constraint matrix.
  Matrix6 C;                                      // XXX: can be precomputed once!
  C.fill(0.f);
  C(0, 2) = 2; C(1, 1) = -1; C(2, 0) = 2;
  
  // Solves the generalized EV problem S*a = l*C*a
  // eigenvectors() method in Eigen::GeneralizedEigenSolver is commented-out, and
  // both of the matrices are symmetric (also self-adjoint, since they're real),
  // so we use the following method
  // http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html
  // EVs of a self-adjoint matrix are always real.
  
  Eigen::SelfAdjointEigenSolver<Matrix6> esC(C);  // XXX: can be precomputed once!
  auto cLambdaV = esC.eigenvalues();
  for (int i = 0; i < 6; ++i)
  if (cLambdaV(i) > 0)
    cLambdaV(i) = 1 / sqrt(cLambdaV(i));
  
  Eigen::SelfAdjointEigenSolver<Matrix6> esS(S);

  auto phi = esC.eigenvectors() * cLambdaV.asDiagonal() * esS.eigenvectors(); // soln vectors
  auto lam = phi.transpose() * S * phi;                                       // soln vals on diagonal
  
  // Theorem 1: there's a single positive EV.
  int iev = -1;
  for (int i = 0; i < 6; ++i)
  if (lam(i, i) > 16*FLT_EPSILON) { // Some non-singularity tolerance
    iev = i;
    break;
  }
  
  if (iev < 0)  // Negative radii will throw exception
    e.setParameters(Point2d<Eigen::Vector3f>(0, 0), -1, -1, -1);
  
  // Find mu, eq.(9) in Fitzgibbon's paper
  auto U = phi(iev);
  float mu = sqrt(U.transpose() * S * U); // XXX: can use C instead of S, more numerically stable?

  if (fabs(mu) < 16*FLT_EPSILON)
    e.setParameters(Point2d<Eigen::Vector3f>(0, 0), -1, -1, -1);
    
  auto A = U / mu;
  
  // A contains _already scaled_ coefs of the ellipse eq, following code is copy-pasted from opencv
  
  
}

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

