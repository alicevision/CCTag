/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/CCTag.hpp>
#include <cctag/utils/Defines.hpp>
#include <cctag/DataSerialization.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/algebra/matrix/Operation.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/2DTransform.hpp>

#include <opencv2/core/core_c.h>

#include <boost/foreach.hpp>
#include <boost/array.hpp>
#include <boost/mpl/bool.hpp>

#include <cstddef>
#include <cmath>
#include <iomanip>

namespace cctag
{

namespace optimization = cctag::numerical::optimization;

const boost::array<float, 5> CCTag::_radiusRatiosInit =
{
  (29.0f / 9.0f),
  (29.0f / 13.0f),
  (29.0f / 17.0f),
  (29.0f / 21.0f),
  (29.0f / 25.0f)
};

bool CCTag::isEqual(const CCTag& marker) const
{
  using namespace cctag::numerical::geometry;
  
  Ellipse centerEllipseA = _rescaledOuterEllipse;
  centerEllipseA.setA( centerEllipseA.b()*0.5f );
  centerEllipseA.setB( centerEllipseA.b()*0.5f );
  
  Ellipse centerEllipseB = marker.rescaledOuterEllipse();
  centerEllipseB.setA( centerEllipseB.b()*0.5f );
  centerEllipseB.setB( centerEllipseB.b()*0.5f );
  
  //bool sameSemiAxis =
  //          ( std::abs( _rescaledOuterEllipse.a()/marker.rescaledOuterEllipse().a() - 1 ) < 0.3 ) &&
  //          ( std::abs( _rescaledOuterEllipse.b()/marker.rescaledOuterEllipse().b() - 1 ) < 0.3 );
  
  return isOverlappingEllipses(centerEllipseA, centerEllipseB);// && sameSemiAxis;
}

void CCTag::condition(const Eigen::Matrix3f & mT, const Eigen::Matrix3f & mInvT)
{
  using namespace cctag::numerical::geometry;

  // Condition outer ellipse
  _outerEllipse = _outerEllipse.transform(mInvT);
  cctag::numerical::normalizeDet1(_outerEllipse.matrix());

  // Condition all ellipses
  for(cctag::numerical::geometry::Ellipse & ellipse : _ellipses)
  {
    ellipse = ellipse.transform(mInvT);
    cctag::numerical::normalizeDet1(ellipse.matrix());
  }

  for(std::vector<cctag::DirectedPoint2d<Eigen::Vector3f> > & points : _points)
  {
    cctag::numerical::optimization::condition(points, mT);
  }

  cctag::numerical::optimization::condition(_centerImg, mT);
}

void CCTag::applyScale(float s)
{

  for(std::vector< DirectedPoint2d<Eigen::Vector3f> > &vp : _points)
  {

    for(DirectedPoint2d<Eigen::Vector3f> & p : vp)
    {
      p.x() = p.x() * s;
      p.y() = p.y() * s;
    }
  }

  _centerImg.x() = _centerImg.x() * s;
  _centerImg.y() = _centerImg.y() * s;
  
  _outerEllipse.setCenter(Point2d<Eigen::Vector3f>(_outerEllipse.center().x() * s,
                          _outerEllipse.center().y() * s));
  _outerEllipse.setA(_outerEllipse.a() * s);
  _outerEllipse.setB(_outerEllipse.b() * s);
}

#ifdef CCTAG_WITH_CUDA
void CCTag::acquireNearbyPointMemory( int tagId )
{
    _cuda_result = cctag::PinnedCounters::getPointPtr( tagId, __FILE__, __LINE__ );
}

void CCTag::releaseNearbyPointMemory( int tagId )
{
    cctag::PinnedCounters::releaseAllPoints( tagId );
}
#endif

void CCTag::serialize(boost::archive::text_oarchive & ar, unsigned int version)
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
  serializeMatrix3f(ar, _mHomography);
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
	 << "    (" << _centerImg.x() << "," << _centerImg.y() << ")" << endl
         << "    nCircles: " << _nCircles << endl
	 << "    radius ratios: ";
    for( float x : _radiusRatios ) {
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
    for( const std::vector< DirectedPoint2d<Eigen::Vector3f> >& v : _points ) {
	ostr << "        ";
    	for( const DirectedPoint2d<Eigen::Vector3f>& p : v ) {
            ostr << "(" << p.x() << "," << p.y() << ") ";
    	}
	ostr << endl;
    }
}
#endif
} // namespace cctag
