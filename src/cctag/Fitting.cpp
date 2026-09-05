/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/utils/Defines.hpp>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/Fitting.hpp>
#include <cmath>
#include <cfloat>
#include <limits>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <Eigen/Eigenvalues>

namespace cctag {
namespace numerical {

namespace geometry
{

// The direct least-squares ellipse fit (Halir and Flusser) runs in a
// normalised frame and in double precision: the points are translated to
// their centroid and divided by their RMS radius, so every scatter-matrix
// entry is O(n) whatever the pixel scale. In the raw pixel frame the
// quadratic scatter entries reach 1e12 for an arc a few hundred pixels
// across, past the 24-bit float mantissa, and whether a candidate ellipse
// assembles then depends on the rounding regime of the host (fused
// multiply-add on AArch64, separate multiply and add on baseline x86-64).
using Vector6d = Eigen::Matrix<double, 6, 1>;
// 6 conic coefficients in the centred pixel frame + the centroid offset.
using Conic = std::tuple<Vector6d, Eigen::Vector2d>;

// Relative conditioning floor of the linear scatter block: det / trace^2 of
// the centred point covariance, which tends to lambda_min / lambda_max for a
// collapsed arc. 1e-12 keeps four significant digits in double and rejects
// only point sets that are collinear to within rounding; the geometric
// coverage gates belong to the callers.
static constexpr double kScatterConditioningFloor = 1e-12;

template<typename It>
static Eigen::Vector2d get_offset(It begin, It end)
{
  Eigen::Vector2d center(0.0, 0.0);
  const size_t n = end - begin;
  for (; begin != end; ++begin)
    center += Eigen::Vector2d(static_cast<double>((*begin)(0)), static_cast<double>((*begin)(1)));
  return center / static_cast<double>(n);
}

template<typename It>
static double get_scale(It begin, It end, const Eigen::Vector2d& offset)
{
  double acc = 0.0;
  size_t n = 0;
  for (; begin != end; ++begin, ++n)
  {
    const Eigen::Vector2d p(static_cast<double>((*begin)(0)), static_cast<double>((*begin)(1)));
    acc += (p - offset).squaredNorm();
  }
  const double scale = n > 0 ? std::sqrt(acc / static_cast<double>(n)) : 0.0;
  return (std::isfinite(scale) && scale > 0.0) ? scale : 1.0;
}

template<typename It>
static std::tuple<Eigen::Matrix3d,Eigen::Matrix3d,Eigen::Matrix3d>
get_scatter_matrix(It begin, It end, const Eigen::Vector2d& offset, double scale)
{
  using namespace Eigen;

  const size_t n = end - begin;
  MatrixX3d D1(n,3), D2(n,3);

  // Construct the quadratic and linear parts in the normalised frame.
  for (size_t i = 0; begin != end; ++begin, ++i)
  {
    const Vector2d p(static_cast<double>((*begin)(0)), static_cast<double>((*begin)(1)));
    const Vector2d pc = (p - offset) / scale;
    D1.row(i) = Vector3d(pc(0)*pc(0), pc(0)*pc(1), pc(1)*pc(1));
    D2.row(i) = Vector3d(pc(0), pc(1), 1.0);
  }

  // Construct the three parts of the symmetric scatter matrix.
  Matrix3d S1 = D1.transpose() * D1;
  Matrix3d S2 = D1.transpose() * D2;
  Matrix3d S3 = D2.transpose() * D2;
  return std::make_tuple(S1, S2, S3);
}

template<typename It>
static Conic fit_solver(It begin, It end)
{
  using namespace Eigen;
  using std::get;

  static const struct C1_Initializer {
    Matrix3d inverse;
    C1_Initializer()
    {
      inverse <<
            0,  0, 0.5,
            0, -1,   0,
          0.5,  0,   0;
    };
  } C1;

  const Vector2d offset = get_offset(begin, end);
  const double scale = get_scale(begin, end, offset);
  const auto St = get_scatter_matrix(begin, end, offset, scale);
  const Matrix3d& S1 = std::get<0>(St);
  const Matrix3d& S2 = std::get<1>(St);
  const Matrix3d& S3 = std::get<2>(St);

  // S3 is the scatter of (x, y, 1); centred, its (x, y) block is n times the
  // point covariance and its last diagonal entry is n. A relative test on
  // that block replaces an absolute determinant threshold, which any
  // pixel-scaled scatter passes even when the arc is a straight segment.
  const double trace_xy = S3(0,0) + S3(1,1);
  const double det_xy = S3(0,0) * S3(1,1) - S3(0,1) * S3(1,0);
  if (!S3.allFinite() || !(S3(2,2) > 0.0) || !(trace_xy > 0.0) ||
      !(det_xy > kScatterConditioningFloor * trace_xy * trace_xy))
  {
      throw std::domain_error("fit_solver: the input points appear to be linearly dependent");
  }
  const Matrix3d T = -S3.fullPivLu().solve(S2.transpose());
  const Matrix3d M = C1.inverse * (S1 + S2*T);
  if (!M.allFinite())
  {
      throw std::domain_error("fit_solver: the reduced scatter matrix is not finite");
  }

  EigenSolver<Matrix3d> M_ev(M);
  if (M_ev.info() != Success)
  {
      throw std::domain_error("fit_solver: the eigensolver did not converge");
  }
  const Matrix3d evr = M_ev.eigenvectors().real();
  const Vector3d cond =
      (4.0 * evr.row(0).array() * evr.row(2).array() - evr.row(1).array() * evr.row(1).array()).transpose();

  // The ellipse solution is the eigenvector whose constraint value 4ac - b^2
  // is positive; exactly one exists in exact arithmetic. The float epsilon
  // stays the positivity margin on the unit-norm eigenvectors.
  const double eps = std::numeric_limits<float>::epsilon();
  double minValue = std::numeric_limits<double>::max();
  int imin = -1;
  for (int i = 0; i < 3; ++i)
  {
      if (cond(i) > eps && cond(i) < minValue)
      {
          imin = i;
          minValue = cond(i);
      }
  }
  if (imin == -1)
  {
      throw std::domain_error("fit_solver: degeneracy");
  }
  const Vector3d a1 = evr.col(imin);
  const Vector3d a2 = T * a1;
  // Back to the centred pixel frame X = x - offset: with x' = X / scale the
  // conic a x'^2 + b x'y' + c y'^2 + d x' + e y' + f = 0 reads, after
  // multiplying through by scale^2, (a, b, c, d scale, e scale, f scale^2).
  Vector6d ret;
  ret << a1(0), a1(1), a1(2), a2(0) * scale, a2(1) * scale, a2(2) * scale * scale;
  return std::make_tuple(ret, offset);
}

// Adapted from OpenCV old code; see
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
void to_ellipse(const Conic& conic, Ellipse& ellipse)
{
  using namespace Eigen;
  // The float epsilon stays the margin of every degeneracy test below: the
  // conic coefficients are O(1) in the normalised fit, so the margins keep
  // the meaning they have on unit-scale data.
  const double eps = std::numeric_limits<float>::epsilon();

  Vector6d coef = std::get<0>(conic);

  double idet = coef(0)*coef(2) - coef(1)*coef(1)/4; // ac-b^2/4
  idet = idet > eps ? 1.0/idet : 0.0;

  const double scale = std::sqrt(idet/4);
  if (!(scale >= eps))
  {
      throw std::domain_error("to_ellipse_2: singularity 1");
  }

  coef *= scale;
  const double aa = coef(0), bb = coef(1), cc = coef(2), dd = coef(3), ee = coef(4);
  double ff = coef(5);

  const Vector2d c = Vector2d(-dd*cc + ee*bb/2, -aa*ee + dd*bb/2) * 2;

  // offset ellipse to (x0,y0)
  ff += aa*c(0)*c(0) + bb*c(0)*c(1) + cc*c(1)*c(1) + dd*c(0) + ee*c(1);
  if (!(std::fabs(ff) >= eps))
  {
      throw std::domain_error("to_ellipse_2: singularity 2");
  }

  Matrix2d S;
  S << aa, bb/2, bb/2, cc;
  S /= -ff;

  // SVs are sorted from largest to smallest
  JacobiSVD<Matrix2d> svd(S, ComputeFullU);
  const auto& vals = svd.singularValues();
  const auto& mat_u = svd.matrixU();

  const Vector2d center = c + std::get<1>(conic);
  if (!(vals(0) > 0.0) || !(vals(1) > 0.0))
  {
	  throw std::domain_error("Degenerate ellipse after fitEllipse => line or point.");
  }
  const Vector2d radius(std::sqrt(1.0/vals(0)), std::sqrt(1.0/vals(1)));
  const double angle = boost::math::constants::pi<double>() - std::atan2(mat_u(0,1), mat_u(1,1));

  if (!(radius(0) > 0.0) || !(radius(1) > 0.0) || !radius.allFinite() || !center.allFinite())
  {
	  throw std::domain_error("Degenerate ellipse after fitEllipse => line or point.");
  }

  ellipse.setParameters(Point2d<Eigen::Vector3f>(static_cast<float>(center(0)), static_cast<float>(center(1))),
                        static_cast<float>(radius(0)), static_cast<float>(radius(1)), static_cast<float>(angle));
}

template<typename It>
void fitEllipse(It begin, It end, Ellipse& e)
{
    const auto numPts = std::distance(begin, end);
    if(numPts < 5)
    {
        std::cout << "fitEllipse it: " + std::to_string(numPts) + " provided, at least 5 are needed to estimate an ellipse" << std::endl;
        throw std::domain_error(
                "fitEllipse: " + std::to_string(numPts) + " provided, at least 5 are needed to estimate an ellipse");
    }

    auto conic = fit_solver(begin, end);
    geometry::to_ellipse(conic, e);
}

// explicit instantiations
template void fitEllipse(std::vector<cctag::Point2d<Eigen::Vector3f>>::const_iterator begin,
  std::vector<cctag::Point2d<Eigen::Vector3f>>::const_iterator end, Ellipse& e);

} // geometry

float innerProdMin(const std::vector<cctag::EdgePoint*>& filteredChildren, float thrCosDiffMax, Point2d<Vector3s> & p1, Point2d<Vector3s> & p2) {

            EdgePoint* pAngle1 = nullptr;
            EdgePoint* pAngle2 = nullptr;

            float min = 1.1f;

            float distMax = 0.f;

            if (!filteredChildren.empty())
            {
                EdgePoint* p0 = filteredChildren.front();

                float normGrad = std::sqrt(p0->dX() * p0->dX() + p0->dY() * p0->dY());

                // Step 1
                float gx0 = p0->dX() / normGrad;
                float gy0 = p0->dY() / normGrad;

                std::vector<cctag::EdgePoint*>::const_iterator it = ++filteredChildren.begin();

                for (; it != filteredChildren.end(); ++it) {
                    EdgePoint* pCurrent = *it;

                    normGrad = std::sqrt(pCurrent->dX() * pCurrent->dX() + pCurrent->dY() * pCurrent->dY());

                    float gx = pCurrent->dX() / normGrad;
                    float gy = pCurrent->dY() / normGrad;

                    float innerProd = gx0 * gx + gy0 * gy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle1 = pCurrent;
                    }

                    float dist = cctag::numerical::distancePoints2D(*p0, *pCurrent);
                    if (dist > distMax) {
                        distMax = dist;
                        p1 = *pCurrent;
                    }
                }

                normGrad = std::sqrt(pAngle1->dX() * pAngle1->dX() + pAngle1->dY() * pAngle1->dY());
                float gxmin = pAngle1->dX() / normGrad;
                float gymin = pAngle1->dY() / normGrad;

                // Step 2, compute the minimum inner product
                min = 1.f;
                distMax = 0.f;

                it = filteredChildren.begin();

                //CCTAG_COUT(" 2- 2eme element" << **it);

                for (; it != filteredChildren.end(); ++it) {
                    EdgePoint* pCurrent = *it;

                    normGrad = std::sqrt(pCurrent->dX() * pCurrent->dX() + pCurrent->dY() * pCurrent->dY());

                    float chgx = pCurrent->dX() / normGrad;
                    float chgy = pCurrent->dY() / normGrad;

                    float innerProd = gxmin * chgx + gymin * chgy;

                    if (innerProd <= thrCosDiffMax)
                        return innerProd;

                    if (innerProd < min) {
                        min = innerProd;
                        pAngle2 = pCurrent;
                    }

                    float dist = cctag::numerical::distancePoints2D(p1, (Point2d<Vector3s>)(*pCurrent));
                    if (dist > distMax) {
                        distMax = dist;
                        p2 = *pCurrent;
                    }
                }
            }

            return min;
        }


void ellipseFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<Point2d<Eigen::Vector3f>>& points)
{
  geometry::fitEllipse(points.begin(), points.end(), e);
}

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points )
{
    const auto numPts = points.size();
    if(numPts < 5)
    {
        std::cout << "fitEllipse: " + std::to_string(numPts) + " provided, at least 5 are needed to estimate an ellipse" << std::endl;
        throw std::domain_error(
                "fitEllipse: " + std::to_string(numPts) + " provided, at least 5 are needed to estimate an ellipse");
    }
    using indirect_iterator = boost::indirect_iterator<std::vector<cctag::EdgePoint*>::const_iterator>;
    geometry::fitEllipse(indirect_iterator(points.begin()), indirect_iterator(points.end()), e);
}

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points) {

  const std::size_t nPoints = points.size();
  if (nPoints < 3)
  {
	  throw std::domain_error("circleFitting: " + std::to_string(nPoints) + " provided, at least 3 are needed to estimate a circle");
  }

  // The algebraic circle fit is the null vector of [x y 1 x^2+y^2]; in the
  // raw pixel frame its columns differ by the square of the pixel scale, so
  // the points are centred and divided by their RMS radius first, as in the
  // ellipse fit, and the solve runs in double precision.
  Eigen::Vector2d offset(0.0, 0.0);
  for (std::size_t i = 0; i < nPoints; ++i)
      offset += Eigen::Vector2d(points[i]->x(), points[i]->y());
  offset /= static_cast<double>(nPoints);
  double acc = 0.0;
  for (std::size_t i = 0; i < nPoints; ++i)
      acc += (Eigen::Vector2d(points[i]->x(), points[i]->y()) - offset).squaredNorm();
  double scale = std::sqrt(acc / static_cast<double>(nPoints));
  if (!(std::isfinite(scale) && scale > 0.0))
      scale = 1.0;

  Eigen::MatrixXd A(nPoints, 4);
  for (std::size_t i = 0; i < nPoints; ++i) {
      const Eigen::Vector2d pc = (Eigen::Vector2d(points[i]->x(), points[i]->y()) - offset) / scale;
      A(i, 0) = pc(0);
      A(i, 1) = pc(1);
      A(i, 2) = 1.0;
      A(i, 3) = pc.squaredNorm();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto V = svd.matrixV();

  // A vanishing quadratic coefficient is a line, not a circle.
  if (!(std::fabs(V(3, 3)) > std::numeric_limits<double>::epsilon()))
  {
	  throw std::domain_error("Degenerate circle in circleFitting, the points are collinear");
  }
  const double xC = -0.5 * V(0, 3) / V(3, 3);
  const double yC = -0.5 * V(1, 3) / V(3, 3);
  const double radius2 = xC*xC + yC*yC - V(2, 3) / V(3, 3);

  if (!(radius2 > 0.0) || !std::isfinite(radius2))
  {
	  throw std::domain_error("Degenerate circle in circleFitting, squared radius is not positive or not finite: " + std::to_string(radius2));
  }

  const double radius = std::sqrt(radius2) * scale;
  const Eigen::Vector2d center = offset + Eigen::Vector2d(xC, yC) * scale;
  e.setParameters(Point2d<Eigen::Vector3f>(static_cast<float>(center(0)), static_cast<float>(center(1))),
                  static_cast<float>(radius), static_cast<float>(radius), 0);
}

} // namespace numerical
} // namespace cctag
