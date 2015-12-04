#include <cctag/CCTag.hpp>
#include <cctag/global.hpp>
#include <cctag/dataSerialization.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/optimization/conditioner.hpp>
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

namespace cctag
{

namespace ublas = boost::numeric::ublas;
namespace optimization = cctag::numerical::optimization;

// todo@Lilian : used in the initRadiusRatio called in the CCTag constructor. Need to be changed while reading the CCTagBank build from the textFile.
const boost::array<double, 5> CCTag::_radiusRatiosInit =
{
  (29.0 / 9.0),
  (29.0 / 13.0),
  (29.0 / 17.0),
  (29.0 / 21.0),
  (29.0 / 25.0)
};

void CCTag::condition(const cctag::numerical::BoundedMatrix3x3d & mT, const cctag::numerical::BoundedMatrix3x3d & mInvT)
{
  using namespace cctag::numerical::geometry;

  // Condition outer ellipse
  _outerEllipse = _outerEllipse.transform(mInvT);
  cctag::numerical::normalizeDet1(_outerEllipse.matrix());

  // Condition all ellipses
  BOOST_FOREACH(cctag::numerical::geometry::Ellipse & ellipse, _ellipses)
  {
    ellipse = ellipse.transform(mInvT);
    cctag::numerical::normalizeDet1(ellipse.matrix());
  }

  BOOST_FOREACH(std::vector<cctag::DirectedPoint2d<double> > & points, _points)
  {
    cctag::numerical::optimization::condition(points, mT);
  }

  cctag::numerical::optimization::condition(_centerImg, mT);
}

void CCTag::scale(const double s)
{

  BOOST_FOREACH(std::vector< DirectedPoint2d<double> > &vp, _points)
  {

    BOOST_FOREACH(DirectedPoint2d<double> & p, vp)
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

#ifdef WITH_CUDA
void CCTag::acquireNearbyPointMemory( )
{
    _cuda_result = popart::PinnedCounters::getPointPtr();
}

void CCTag::releaseNearbyPointMemory( )
{
    popart::PinnedCounters::releaseAllPoints();
}
#endif

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
#ifdef CCTAG_SERIALIZE
  serializeFlowComponents(ar, _flowComponents);
#endif
}

#ifndef NDEBUG
using namespace std;

void CCTag::printTag( std::ostream& ostr ) const
{
    ostr << setprecision(4)
         << "CCTag:" << endl
	 << "    (" << _centerImg.getX() << "," << _centerImg.getY() << ")" << endl
         << "    nCircles: " << _nCircles << endl
	 << "    radius ratios: ";
    for( double x : _radiusRatios ) {
        ostr << x << " ";
    }
    ostr << endl
	 << "    ID: " << _id << endl
	 << "    quality: " << _quality << endl
	 << "    level: " << _pyramidLevel << endl
	 << "    scale: " << _scale << endl
	 << "    status: " << _status << endl
         << "    outerEllipse: " << _outerEllipse << endl
         << "    mHomography: " << "[ " << _mHomography(0,0) << " " << _mHomography(0,1) << " " << _mHomography(0,2) << " ; "
		                << _mHomography(1,0) << " " << _mHomography(1,1) << " " << _mHomography(1,2) << " ; "
			        << _mHomography(2,0) << " " << _mHomography(2,1) << " " << _mHomography(2,2) << " ] " << endl
	 << "    Ellipses: " << endl;
    for( const cctag::numerical::geometry::Ellipse& e : _ellipses ) {
        ostr << "        " << e << endl;
    }
    ostr << "    rescaledOuterEllipse: " << _rescaledOuterEllipse << endl
         << "    Points: " << endl;
    for( const std::vector< DirectedPoint2d<double> >& v : _points ) {
	ostr << "        ";
    	for( const DirectedPoint2d<double>& p : v ) {
            ostr << "(" << p.x() << "," << p.y() << ") ";
    	}
	ostr << endl;
    }
}
#endif
} // namespace cctag
