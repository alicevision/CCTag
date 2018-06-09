/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/utils/Defines.hpp>
#include <Eigen/SVD>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <cctag/geometry/Point.hpp>
#include <cctag/Fitting.hpp>
#include <cmath>
#include <cfloat>
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

using Vector6f = Eigen::Matrix<float, 6, 1>;
using Conic = std::tuple<Vector6f, Eigen::Vector2f>; // 6 coefficients + center offset

template<typename It>
static Eigen::Vector2f get_offset(It begin, It end)
{
  Eigen::Vector2f center(0, 0);
  const size_t n = end - begin;
  for (; begin != end; ++begin)
    center += Eigen::Vector2f((*begin)(0), (*begin)(1));
  return center / n;
}

template<typename It>
static std::tuple<Eigen::Matrix3f,Eigen::Matrix3f,Eigen::Matrix3f>
get_scatter_matrix(It begin, It end, const Eigen::Vector2f& offset)
{
  using namespace Eigen;
  
  const auto qf = [&](It it) {
    Vector2f p((*it)(0), (*it)(1));
    auto pc = p - offset;
    return Vector3f(pc(0)*pc(0), pc(0)*pc(1), pc(1)*pc(1));
  };
  const auto lf = [&](It it) {
    Vector2f p((*it)(0), (*it)(1));
    auto pc = p - offset;
    return Vector3f(pc(0), pc(1), 1);
  };
  
  const size_t n = end - begin;
  MatrixX3f D1(n,3), D2(n,3);
  
  // Construct the quadratic and linear parts.  Doing it in two loops has better cache locality.
  // TODO@stian: Make an nx6 matrix and define D1 and D2 as subblocks so that they're as one memory block
  for (size_t i = 0; begin != end; ++begin, ++i) {
    D1.row(i) = qf(begin);
    D2.row(i) = lf(begin);
  }
  
  // Construct the three parts of the symmetric scatter matrix.
  Matrix3f S1 = D1.transpose() * D1;
  Matrix3f S2 = D1.transpose() * D2;
  Matrix3f S3 = D2.transpose() * D2;
  return std::make_tuple(S1, S2, S3);
}

template<typename It>
static Conic fit_solver(It begin, It end)
{
  using namespace Eigen;
  using std::get;
  
  static const struct C1_Initializer {
    Matrix3f matrix;
    Matrix3f inverse;
    C1_Initializer()
    {
      matrix <<
          0,  0, 2,
          0, -1, 0,
          2,  0, 0;
      inverse <<
            0,  0, 0.5,
            0, -1,   0,
          0.5,  0,   0; 
    };
  } C1;
  
  const auto offset = get_offset(begin, end);
  const auto St = get_scatter_matrix(begin, end, offset);
  const auto& S1 = std::get<0>(St);
  const auto& S2 = std::get<1>(St);
  const auto& S3 = std::get<2>(St);
  const auto T = -S3.inverse() * S2.transpose();
  const auto M = C1.inverse * (S1 + S2*T);
  
  EigenSolver<Matrix3f> M_ev(M);
  Vector3f cond;
  {
    const auto evr = M_ev.eigenvectors().real().array();
    cond = 4*evr.row(0)*evr.row(2) - evr.row(1)*evr.row(1);
  }

  float min = FLT_MAX;
  int imin = -1;
  for (int i = 0; i < 3; ++i)
  if (cond(i) > 0 && cond(i) < min) {
    imin = i; min = cond(i);
  }
  
  Vector6f ret = Matrix<float, 6, 1>::Zero();
  if (imin >= 0) {
    Vector3f a1 = M_ev.eigenvectors().real().col(imin);
    Vector3f a2 = T*a1;
    ret.block<3,1>(0,0) = a1;
    ret.block<3,1>(3,0) = a2;
  }
  return std::make_tuple(ret, offset);
}

// Adapted from OpenCV old code; see
// https://github.com/Itseez/opencv/commit/4eda1662aa01a184e0391a2bb2e557454de7eb86#diff-97c8133c3c171e64ea0df0db4abd033c
void to_ellipse(const Conic& conic, Ellipse& ellipse)
{
  using namespace Eigen;
  auto coef = std::get<0>(conic);

  float idet = coef(0)*coef(2) - coef(1)*coef(1)/4; // ac-b^2/4
  idet = idet > FLT_EPSILON ? 1.f/idet : 0;
  
  float scale = std::sqrt(idet/4);
  if (scale < FLT_EPSILON)
    throw std::domain_error("to_ellipse_2: singularity 1");
  
  coef *= scale;
  float aa = coef(0), bb = coef(1), cc = coef(2), dd = coef(3), ee = coef(4), ff = coef(5);
  
  const Vector2f c = Vector2f(-dd*cc + ee*bb/2, -aa*ee + dd*bb/2) * 2;
  
  // offset ellipse to (x0,y0)
  ff += aa*c(0)*c(0) + bb*c(0)*c(1) + cc*c(1)*c(1) + dd*c(0) + ee*c(1);
  if (std::fabs(ff) < FLT_EPSILON)
   throw std::domain_error("to_ellipse_2: singularity 2");
  
  Matrix2f S;
  S << aa, bb/2, bb/2, cc;
  S /= -ff;

  // SVs are sorted from largest to smallest
  JacobiSVD<Matrix2f> svd(S, ComputeFullU);
  const auto& vals = svd.singularValues();
  const auto& mat_u = svd.matrixU();

  Vector2f center = c + std::get<1>(conic);
  Vector2f radius = Vector2f(std::sqrt(1.f/vals(0)), std::sqrt(1.f/vals(1)));
  float angle = boost::math::constants::pi<float>() - std::atan2(mat_u(0,1), mat_u(1,1));
  
  if (radius(0) <= 0 || radius(1) <= 0)
    CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate ellipse after fitEllipse => line or point."));
  
  ellipse.setParameters(Point2d<Eigen::Vector3f>(center(0), center(1)), radius(0), radius(1), angle);
}

template<typename It>
void fitEllipse(It begin, It end, Ellipse& e)
{
  auto conic = fit_solver(begin, end);
  geometry::to_ellipse(conic, e);
}

// explicit instantiations
template void fitEllipse(std::vector<cctag::Point2d<Eigen::Vector3f>>::const_iterator begin,
  std::vector<cctag::Point2d<Eigen::Vector3f>>::const_iterator end, Ellipse& e);

} // geometry

float innerProdMin(const std::vector<cctag::EdgePoint*>& filteredChildren, float thrCosDiffMax, Point2d<Vector3s> & p1, Point2d<Vector3s> & p2) {
            using namespace boost::numeric;
            //using namespace cctag::numerical;

            EdgePoint* pAngle1 = nullptr;
            EdgePoint* pAngle2 = nullptr;

            float min = 1.1f;

            float distMax = 0.f;

            EdgePoint* p0 = filteredChildren.front();

            if (!filteredChildren.empty())
            {

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
  using indirect_iterator = boost::indirect_iterator<std::vector<cctag::EdgePoint*>::const_iterator>;
  geometry::fitEllipse(indirect_iterator(points.begin()), indirect_iterator(points.end()), e);
}

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points) {
  using namespace boost::numeric;

  std::size_t nPoints = points.size();

  Eigen::MatrixXf A(nPoints, 4);

  for (int i = 0; i < nPoints; ++i) {
      A(i, 0) = points[i]->x();
      A(i, 1) = points[i]->y();
      A(i, 2) = 1;
      A(i, 3) = points[i]->x() * points[i]->x() + points[i]->y() * points[i]->y();
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto V = svd.matrixV();

  float xC = -0.5f * V(0, 3) / V(3, 3);
  float yC = -0.5f * V(1, 3) / V(3, 3);
  float radius = sqrt(xC*xC + yC*yC - V(2, 3) / V(3, 3));

  if (radius <= 0) {
      CCTAG_THROW(exception::BadHandle() << exception::dev("Degenerate circle in circleFitting."));
  }

  e.setParameters(Point2d<Eigen::Vector3f>(xC, yC), radius, radius, 0);
}

} // namespace numerical
} // namespace cctag
