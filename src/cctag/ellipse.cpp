#include "ellipse.hpp"
#include "CCTag.hpp"
#include "EdgePoint.hpp"
#include "toolbox.hpp"
#include "visualDebug.hpp"
#include "fileDebug.hpp"
#include "ellipseFittingWithGradient.hpp"

#include <cctag/geometry/Cercle.hpp>
#include <cctag/geometry/point.hpp>
#include <cctag/global.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>

#include <boost/math/special_functions/pow.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/foreach.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/multi_array.hpp>
#include <boost/multi_array/subarray.hpp>
#include <boost/assert.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cctag
{
namespace vision
{
namespace marker
{

bool initMarkerCenter(cctag::Point2dN<double> & markerCenter,
        const std::vector< std::vector< Point2dN<double> > > & markerPoints,
        int realPixelPerimeter)
{
  cctag::numerical::geometry::Ellipse innerEllipse;
  std::size_t nbEllipse = markerPoints.size();

  try
  {
    if (realPixelPerimeter > 200)
    {
      if (markerPoints[0].size() > 20)
      {
        numerical::ellipseFitting(innerEllipse, markerPoints[0]);

        BOOST_FOREACH(Point2dN<double> pt, markerPoints[0])
        {
          CCTagVisualDebug::instance().drawPoint(pt, cctag::color_red);
        }
      }
      else
      {
        return false;
      }
    }
    else
    {
      return false;
    }
  }
  catch (...)
  {
    return false;
  }

  markerCenter = innerEllipse.center();

  return true;
}

bool addCandidateFlowtoCCTag(const std::vector< EdgePoint* > & filteredChildrens,
        const std::vector< EdgePoint* > & outerEllipsePoints,
        const cctag::numerical::geometry::Ellipse& outerEllipse,
        std::vector< std::vector< Point2dN<double> > >& cctagPoints,
        std::size_t numCircles)
{
  using namespace boost::numeric::ublas;

  //cctag::numerical::geometry::Ellipse innerBoundEllipse(outerEllipse.center(), outerEllipse.a()/8.0, outerEllipse.b()/8.0, outerEllipse.angle());
  cctagPoints.resize(numCircles);

  std::vector< std::vector< Point2dN<double> > >::reverse_iterator itp = cctagPoints.rbegin();
  itp->reserve(outerEllipsePoints.size());

  BOOST_FOREACH(EdgePoint * e, outerEllipsePoints)
  {
    itp->push_back(Point2dN<double>(e->x(), e->y()));
  }
  ++itp;
  for (; itp != cctagPoints.rend(); ++itp)
  {
    itp->reserve(filteredChildrens.size());
  }

  std::list<EdgePoint*> vProcessedEdgePoint;

  ROM_COUT_VAR_DEBUG(outerEllipse);

  bounded_vector<double, 2> gradE(2);
  bounded_vector<double, 2> toto(2);

  std::size_t nGradientOut = 0;
  std::size_t nAddedPoint = 0;

  for (std::vector<EdgePoint*>::const_iterator it = filteredChildrens.begin(); it != filteredChildrens.end(); ++it)
  {
    int dir = -1;
    EdgePoint* p = *it;
    const Point2dN<double> outerPoint(p->x(), p->y());

    boost::numeric::ublas::bounded_vector<double, 3> lineThroughCenter;
    double a = outerPoint.x() - outerEllipse.center().x();
    double b = outerPoint.y() - outerEllipse.center().y();
    lineThroughCenter(0) = a;
    lineThroughCenter(1) = b;
    lineThroughCenter(2) = -a * outerEllipse.center().x() - b * outerEllipse.center().y();

    for (std::size_t j = 1; j < numCircles; ++j)
    {
      if (dir == -1)
      {
        p = p->_before;
      }
      else
      {
        p = p->_after;
      }


      if (!p->_processedAux)
      {
        //ROM_COUT(*p);

        p->_processedAux = true;
        vProcessedEdgePoint.push_back(p);

        double normGrad = sqrt(p->_grad.x() * p->_grad.x() + p->_grad.y() * p->_grad.y());

        gradE(0) = p->_grad.x() / normGrad;
        gradE(1) = p->_grad.y() / normGrad;

        toto(0) = outerEllipse.center().x() - p->x();
        toto(1) = outerEllipse.center().y() - p->y();

        double distancePointToCenter = sqrt(toto(0) * toto(0) + toto(1) * toto(1));
        toto(0) /= distancePointToCenter;
        toto(1) /= distancePointToCenter;

        Point2dN<double> pointToAdd(p->x(), p->y());

        //ROM_COUT_VAR(double(dir)*inner_prod( gradE, toto ));


        if (isInEllipse(outerEllipse, pointToAdd) && isOnTheSameSide(outerPoint, pointToAdd, lineThroughCenter))
          // isInHull( innerBoundEllipse, outerEllipse, pMid ) && isInHull( innerBoundEllipse, outerEllipse, pointToAdd ) &&
        {
          if ((double(-dir) * inner_prod(gradE, toto) < -0.5) && (j >= numCircles - 2))
          {
            ++nGradientOut;
          }
          cctagPoints[numCircles - j - 1].push_back(pointToAdd);

          if (j >= numCircles - 2)
          {
            ++nAddedPoint;
          }
        }
        else
        {
          CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(PTS_OUT_WHILE_ASSEMBLING);
          cctagPoints.clear();

          BOOST_FOREACH(EdgePoint* point, vProcessedEdgePoint)
          {
            point->_processedAux = false;
          }
          return false;
        }
      }
      dir = -dir;
    }
  }

  BOOST_FOREACH(EdgePoint* point, vProcessedEdgePoint)
  {
    point->_processedAux = false;
  }

  //std::cin.ignore().get();

  if (double(nGradientOut) / double(nAddedPoint) > 0.5)
  {
    cctagPoints.clear();
    CCTagFileDebug::instance().outputFlowComponentAssemblingInfos(BAD_GRAD_WHILE_ASSEMBLING);
    return false;
  }
  else
  {
    return true;
  }
}

/**
 * @brief Check if points are good for ellipse growing.
 *
 * The goal is to avoid particular cases, with a bad initialization. The
 * condition is that the ellipse is too flattened.
 *
 * @pre childrens size >=5
 *
 * @param[in] childrens
 * @param[out] iMin1
 * @param[out] iMin2
 */
bool isGoodEGPoints(const std::vector<EdgePoint*>& filteredChildrens, Point2dN<int> & p1, Point2dN<int> & p2)
{
  BOOST_ASSERT(filteredChildrens.size() >= 5);

  // TODO constante à associer à la classe de l'algorithme... et choisir la meilleure valeur
  static const double thrCosDiffMax = 0.25; //std::cos( boost::math::constants::pi<double>() / 2.0 );

  const double min = numerical::innerProdMin(filteredChildrens, thrCosDiffMax, p1, p2);

  return min <= thrCosDiffMax;
}

/**
 * @brief Create a circle that matches outer ellipse points.
 *
 * @param childrens
 * @param iMin1
 * @param iMin2
 * @return the circle
 */
numerical::geometry::Cercle computeCircleFromOuterEllipsePoints(const std::vector<EdgePoint*>& filteredChildrens, const Point2dN<int> & p1, const Point2dN<int> & p2)
{
  using namespace boost::numeric::ublas;
  // Compute the line passing through filteredChildrens[iMin1] and filteredChildrens[iMin2] and
  // find i such as d(filteredChildrens[i], l) is maximum.
  bounded_matrix<double, 2, 2> mL(2, 2);

  mL(0, 0) = p1.x();
  mL(0, 1) = p1.y();
  mL(1, 0) = p2.x();
  mL(1, 1) = p2.y();

  bounded_vector<double, 2> minones(2);
  minones(0) = -1;
  minones(1) = -1;

  bounded_matrix<double, 2, 2> mLInv(2, 2);

  //Droite l=[a b c] passant par P1 et P2
  //l = inv([x1 y1; x2 y2])*[-1;-1];
  //l = [l;1];

  // TODO inversion d'une matrice 2x2
  cctag::numerical::invert_2x2(mL, mLInv);
  //cctag::toolbox::matrixInvert(mL, mLInv);
  bounded_vector<double, 2> aux = prec_prod(mLInv, minones);
  bounded_vector<double, 3> l(3);
  l(0) = aux(0);
  l(1) = aux(1);
  l(2) = 1;

  const double normL = std::sqrt(boost::math::pow<2>(l(0)) + boost::math::pow<2>(l(1)));

  //double distMax = std::abs( inner_prod( *( filteredChildrens[0] ), l ) ) / normL;

  const EdgePoint * pMax = filteredChildrens.front();
  double distMax = std::min(
                    cctag::numerical::distancePoints2D((Point2dN<int>)(*pMax), p1),
                    cctag::numerical::distancePoints2D((Point2dN<int>)(*pMax), p2));

  double dist;

  BOOST_FOREACH(const EdgePoint * const e, filteredChildrens)
  {
    dist = std::min(
            cctag::numerical::distancePoints2D((Point2dN<int>)(*e), p1),
            cctag::numerical::distancePoints2D((Point2dN<int>)(*e), p2));

    if (dist > distMax)
    {
      distMax = dist;
      pMax = e;
    }
  }

  //ROM_COUT_VAR_DEBUG(std::abs( inner_prod( *( filteredChildrens[iMax] ), l ) ) / normL);

  Point2dN<double> equiPoint;
  double distanceToAdd = cctag::numerical::distancePoints2D(p1, p2) / 50; // match to the max/min of semi-axis ratio for an outer ellipse of a flow candidate

  if (std::abs(inner_prod(*pMax, l)) / normL < 1e-6)
  {
    double normGrad = std::sqrt(pMax->_grad.x() * pMax->_grad.x() + pMax->_grad.y() * pMax->_grad.y());
    double gx = pMax->_grad.x() / normGrad;
    double gy = pMax->_grad.y() / normGrad;
    equiPoint.setX(pMax->x() + distanceToAdd * gx);
    equiPoint.setY(pMax->y() + distanceToAdd * gy);
  }
  else
  {
    equiPoint.setX(pMax->x());
    equiPoint.setY(pMax->y());
  }

  //ROM_COUT("Create a circle \n" << Point2dN<double>(p1->x(), p1->y()) << " \n " << Point2dN<double>(p2->x(), p2->y())
  //	<< " \n" << equiPoint );

  numerical::geometry::Cercle resCircle(Point2dN<double>(p1.x(), p1.y()), Point2dN<double>(p2.x(), p2.y()), equiPoint);
  
  return resCircle;
}

bool ellipseGrowingInit(std::vector<EdgePoint> & points, const std::vector<EdgePoint*>& filteredChildrens, numerical::geometry::Ellipse& ellipse)
{

  Point2dN<int> p1;
  Point2dN<int> p2;

  bool goodInit = true;

  if (isGoodEGPoints(filteredChildrens, p1, p2))
  {
    // Ellipse fitting based on the filtered childrens
    // todo@Lilian create the construction Ellipse(pts) which calls the following.
    numerical::ellipseFitting(ellipse, filteredChildrens);
  }
  else
  {
    // Initialize ellipse to a circle if the ellipse is not covered enough. 
    // Previous call (heuristic based))
    // ellipse = computeCircleFromOuterEllipsePoints(filteredChildrens, p1, p2);
    // Ellipse fitting on the filtered childrens.
    numerical::circleFitting(ellipse, filteredChildrens);

    goodInit = false;
  }

  // todo@Lilian is this loop still required?

  BOOST_FOREACH(EdgePoint & p, points)
  {
    p._processed = -1;
  }

  return goodInit;
}

void connectedPoint(std::vector<EdgePoint*>& pts, const int runId, 
        const EdgePointsImage& img, numerical::geometry::Ellipse& qIn,
        numerical::geometry::Ellipse& qOut, int x, int y)
{
  using namespace boost::numeric::ublas;
  BOOST_ASSERT(img[x][y]);
  img[x][y]->_processed = runId; // Set as processed

  static int xoff[] = {1, 1, 0, -1, -1, -1, 0, 1};
  static int yoff[] = {0, -1, -1, -1, 0, 1, 1, 1};

  for (int i = 0; i < 8; ++i)
  {
    int sx = x + xoff[i];
    int sy = y + yoff[i];
    if (sx >= 0 && sx < int( img.shape()[0]) &&
        sy >= 0 && sy < int( img.shape()[1]))
    {
      EdgePoint* e = img[sx][sy];

      if (e && // If unprocessed
          isInHull(qIn, qOut, e) &&
          e->_processed != runId)
      {
        e->_processed = runId;

        bounded_vector<double, 2> gradE(2);
        gradE(0) = e->_grad.x();
        gradE(1) = e->_grad.y();
        bounded_vector<double, 2> eO(2);
        eO(0) = qIn.center().x() - e->x();
        eO(1) = qIn.center().y() - e->y();

        if (inner_prod(gradE, eO) < 0)
        {
          pts.push_back(e);
          connectedPoint(pts, runId, img, qIn, qOut, sx, sy);
        }
      }
    }
  }
}

void computeHull(const numerical::geometry::Ellipse& ellipse, double delta,
        numerical::geometry::Ellipse& qIn, numerical::geometry::Ellipse& qOut)
{
  qIn = numerical::geometry::Ellipse(cctag::Point2dN<double>(
          ellipse.center().x(),
          ellipse.center().y()),
          std::max(ellipse.a() - delta, 0.001),
          std::max(ellipse.b() - delta, 0.001),
          ellipse.angle());
  qOut = numerical::geometry::Ellipse(cctag::Point2dN<double>(ellipse.center().x(),
          ellipse.center().y()),
          ellipse.a() + delta,
          ellipse.b() + delta,
          ellipse.angle());
}

void ellipseHull(const EdgePointsImage& img,
        std::vector<EdgePoint*>& pts,
        numerical::geometry::Ellipse& ellipse,
        double delta)
{
  numerical::geometry::Ellipse qIn, qOut;
  computeHull(ellipse, delta, qIn, qOut);

  std::size_t initSize = pts.size();

  for (std::size_t i = 0; i < initSize; ++i)
  {
    EdgePoint *e = pts[i];
    connectedPoint(pts, 0, img, qIn, qOut, e->x(), e->y());
  }
}

void ellipseGrowing(const EdgePointsImage& img,
        const std::vector<EdgePoint*>& filteredChildrens, 
        std::vector<EdgePoint*>& outerEllipsePoints,
        numerical::geometry::Ellipse& ellipse,
        const double ellipseGrowingEllipticHullWidth, 
        std::size_t & nSegmentOut,
        std::size_t & nLabel,
        bool goodInit)
{
  outerEllipsePoints.reserve(filteredChildrens.size()*3);

  BOOST_FOREACH(EdgePoint * children, filteredChildrens)
  {
    outerEllipsePoints.push_back(children);
    children->_processed = 0;
  }

  int lastSizePoints = 0;
  // ellipse is initialized by ellipseGrowingInit

  int nIter = 0;

  while (outerEllipsePoints.size() - lastSizePoints > 0)
  {
    lastSizePoints = outerEllipsePoints.size();

    if ((nIter == 0) && (!goodInit))
    {
      ellipseHull(img, outerEllipsePoints, ellipse, 3);
    }
    else
    {
      ellipseHull(img, outerEllipsePoints, ellipse, ellipseGrowingEllipticHullWidth);
    }

    // Compute the new ellipse which fits oulierEllipsePoints
    numerical::ellipseFitting(ellipse, outerEllipsePoints);

    //ROM_TCOUT(ellipse.matrix());

    ++nIter;
  }

  BOOST_FOREACH(EdgePoint * p, outerEllipsePoints)
  {
    p->_processed = -1;
  }
  
}

void ellipseGrowing2(const EdgePointsImage& img,
        const std::vector<EdgePoint*>& filteredChildrens, 
        std::vector<EdgePoint*>& outerEllipsePoints, numerical::geometry::Ellipse& ellipse,
        const double ellipseGrowingEllipticHullWidth,
        std::size_t & nSegmentOut,
        std::size_t & nLabel,
        bool goodInit)
{
  outerEllipsePoints.reserve(filteredChildrens.size()*3);

  BOOST_FOREACH(EdgePoint * children, filteredChildrens)
  {
    outerEllipsePoints.push_back(children);
    children->_processed = 0;
  }

  int lastSizePoints = 0;
  int nIter = 0;

  if (!goodInit)
  {
    int newSizePoints = outerEllipsePoints.size();
    int maxNbPoints = newSizePoints;
    int nIterMax = 0;
    std::vector<std::vector<EdgePoint*> > edgePointsSets;
    edgePointsSets.reserve(6); // maximum of expected iterations
    edgePointsSets.push_back(outerEllipsePoints);
    std::vector<numerical::geometry::Ellipse> ellipsesSets;
    ellipsesSets.reserve(6); // maximum of expected iterations
    ellipsesSets.push_back(ellipse);
    ++nIter;
    while (newSizePoints - lastSizePoints > 0)
    {
      numerical::geometry::Ellipse qIn, qOut;
      computeHull(ellipse, ellipseGrowingEllipticHullWidth, qIn, qOut);
      lastSizePoints = 0;
      BOOST_FOREACH(const EdgePoint * point, outerEllipsePoints)
      {
        if (isInHull(qIn, qOut, point))
        {
          ++lastSizePoints;
        }
      }

      ellipseHull(img, outerEllipsePoints, ellipse, ellipseGrowingEllipticHullWidth);
      edgePointsSets.push_back(outerEllipsePoints);
      ellipsesSets.push_back(ellipse);


      // Compute the new circle which fits oulierEllipsePoints
      numerical::circleFitting(ellipse, outerEllipsePoints);

      computeHull(ellipse, ellipseGrowingEllipticHullWidth, qIn, qOut);
      newSizePoints = 0;
      BOOST_FOREACH(const EdgePoint * point, outerEllipsePoints)
      {
        if (isInHull(qIn, qOut, point))
        {
          ++newSizePoints;
        }
      }
      
      if (newSizePoints > maxNbPoints)
      {
        maxNbPoints = newSizePoints;
        nIterMax = nIter;
      }

      ++nIter;
    }
    outerEllipsePoints = edgePointsSets[nIterMax];
    ellipse = ellipsesSets[nIterMax];
  }

  lastSizePoints = 0;
  nIter = 0;

  // Once the circle is computed, compute the ellipse that fits the same set of points
  numerical::ellipseFitting(ellipse, outerEllipsePoints);

  while (outerEllipsePoints.size() - lastSizePoints > 0)
  {
    lastSizePoints = outerEllipsePoints.size();

    ellipseHull(img, outerEllipsePoints, ellipse, ellipseGrowingEllipticHullWidth);
    // Compute the new ellipse which fits oulierEllipsePoints
    numerical::ellipseFitting(ellipse, outerEllipsePoints);

    ++nIter;
  }

  BOOST_FOREACH(EdgePoint * p, outerEllipsePoints)
  {
    p->_processed = -1;
  }
}

void readPointsFromFile(char* file, std::vector<EdgePoint* >& pts)
{
  using namespace boost::numeric::ublas;
  std::ifstream fid(file, std::ios::in); // on ouvre en lecture

  if (fid) // if file successfully opened
  {
    std::string containt;

    while (getline(fid, containt))
    {
      char* cstr, * p;

      //string str ("Please split this phrase into tokens");

      cstr = new char [containt.size() + 1];
      strcpy(cstr, containt.c_str());

      // cstr now contains a c-string copy of str

      bounded_vector<double, 3> vec;
      vec(2) = 1.f;

      p = strtok(cstr, " ");
      int i = 0;

      while (p != NULL)
      {
        vec(i) = atoi(p);

        ++i;
        p = strtok(NULL, " ");
      }
      ROM_COUT_LILIAN(vec);
      EdgePoint* ePt = new EdgePoint((int) vec(0), (int) vec(1), 1.0f, 1.0f);
      pts.push_back(ePt);

      delete[] cstr;

    }
  }
  else
  {
    std::cerr << "Cannot open the file" << std::endl;

  }
}

void writeMatrix(const CvMat* M, FILE * pFile)
{
  for (int i = 0; i < cvGetSize(M).height; i++)
  {
    for (int j = 0; j < cvGetSize(M).width; j++)
      fprintf(pFile, "%.15f\t", cvmGet(M, i, j));
    fprintf(pFile, "\n");
  }
  fprintf(pFile, "\n");
}

}
}
}


