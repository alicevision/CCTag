#include <cctag/CCTag.hpp>
#include <cctag/global.hpp>
#include <cctag/dataSerialization.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/viewGeometry/homography.hpp>
#include <cctag/viewGeometry/2DTransform.hpp>

#ifdef WITH_CMINPACK
#include <cminpack.h>
#endif

#include <opencv2/core/core_c.h>

#include <boost/foreach.hpp>
#include <boost/timer.hpp>
#include <boost/array.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>

#include <cstddef>
#include <cmath>
#include <iomanip>

namespace popart
{
namespace vision
{
namespace marker
{

namespace ublas = boost::numeric::ublas;
namespace optimization = popart::numerical::optimization;

// todo@Lilian : used in the initRadiusRatio called in the CCTag constructor. Need to be changed while reading the CCTagBank build from the textFile.
const boost::array<double, 5> CCTag::_radiusRatiosInit =
{
  (29.0 / 9.0),
  (29.0 / 13.0),
  (29.0 / 17.0),
  (29.0 / 21.0),
  (29.0 / 25.0)
};

void CCTag::condition(const popart::numerical::BoundedMatrix3x3d & mT, const popart::numerical::BoundedMatrix3x3d & mInvT)
{
  using namespace popart::numerical::geometry;

  // Condition outer ellipse
  _outerEllipse = _outerEllipse.transform(mInvT);
  popart::numerical::normalizeDet1(_outerEllipse.matrix());

  // Condition all ellipses
  BOOST_FOREACH(popart::numerical::geometry::Ellipse & ellipse, _ellipses)
  {
    ellipse = ellipse.transform(mInvT);
    popart::numerical::normalizeDet1(ellipse.matrix());
  }

  BOOST_FOREACH(std::vector<popart::Point2dN<double> > & pts, _points)
  {
    popart::numerical::optimization::condition(pts, mT);
  }

  popart::numerical::optimization::condition(_centerImg, mT);
}

void CCTag::scale(const double s)
{

  BOOST_FOREACH(std::vector< Point2dN<double> > &vp, _points)
  {

    BOOST_FOREACH(Point2dN<double> &p, vp)
    {
      p.setX(p.x() * s);
      p.setY(p.y() * s);
    }
  }

  _centerImg.setX(_centerImg.x() * s);
  _centerImg.setY(_centerImg.y() * s);
  
  _outerEllipse.setCenter(Point2dN<double>(_outerEllipse.center().x() * s,
                          _outerEllipse.center().y() * s));
  _outerEllipse.setA(_outerEllipse.a() * s);
  _outerEllipse.setB(_outerEllipse.b() * s);
}

void CCTag::serialize(boost::archive::text_oarchive & ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(_nCircles);
  ar & BOOST_SERIALIZATION_NVP(_id);
  ar & BOOST_SERIALIZATION_NVP(_pyramidLevel);
  ar & BOOST_SERIALIZATION_NVP(_scale);
  ar & BOOST_SERIALIZATION_NVP(_status);
  serializeEllipse(ar, _outerEllipse);
  serializeEllipse(ar, _rescaledOuterEllipse);
  serializeIdSet(ar, _idSet);
  serializeRadiusRatios(ar, _radiusRatios);
  ar & BOOST_SERIALIZATION_NVP(_quality);
  serializePoints(ar, _points);
  serializeEllipses(ar, _ellipses);
  serializeBoundedMatrix3x3d(ar, _mHomography);
  serializePoint(ar, _centerImg);
#ifdef CCTAG_STAT_DEBUG
  serializeFlowComponents(ar, _flowComponents);
#endif
}

}
}
}
