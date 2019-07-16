/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>

#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/math/constants/constants.hpp>

namespace cctag {
namespace numerical {
namespace geometry {

Point2d<Eigen::Vector3f> extractEllipsePointAtAngle( const Ellipse & ellipse, float theta )
{
    Point2d<Eigen::Vector3f> p;
    float x = ellipse.a() * cos( theta );
    float y = ellipse.b() * sin( theta );
    p.x() = ( x * cos( ellipse.angle() ) - y * sin( ellipse.angle() ) + ellipse.center().x() );
    p.y() = ( x * sin( ellipse.angle() ) + y * cos( ellipse.angle() ) + ellipse.center().y() );
    return p;
}

Point2d<Eigen::Vector3f> pointOnEllipse( const Ellipse & ellipse, const Point2d<Eigen::Vector3f> & p )
{
  Point2d<Eigen::Vector3f> tmp, res;  
  
  // Place into its canonical representation
  float x = p.x() - ellipse.center().x();
  float y = p.y() - ellipse.center().y();

  tmp.x() =    x * cos( ellipse.angle() ) + y * sin( ellipse.angle() );
  tmp.y() =  - x * sin( ellipse.angle() ) + y * cos( ellipse.angle() );
  
  float cs = sqrt( tmp.x()*tmp.x()/(ellipse.a()*ellipse.a()) + tmp.y()*tmp.y()/(ellipse.b()*ellipse.b()) );
  tmp.x() /= cs;
  tmp.y() /= cs;
  
  // Transform back into the original coordinate system
  res.x() = ( tmp.x() * cos( ellipse.angle() ) - tmp.y() * sin( ellipse.angle() ) + ellipse.center().x() );
  res.y() = ( tmp.x() * sin( ellipse.angle() ) + tmp.y() * cos( ellipse.angle() ) + ellipse.center().y() );

  return res;
}

void points( const Ellipse & ellipse, std::size_t nb, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts )
{
	const float step = 2.0f * boost::math::constants::pi<float>() / nb;
	points( ellipse, nb, step, 2 * boost::math::constants::pi<float>(), pts );
}

void points( const Ellipse & ellipse, std::size_t nb, const float phi1, const float phi2, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts )
{
	const float step = 2.0f * boost::math::constants::pi<float>() / nb;
	pts.reserve( std::size_t( ( phi2 - phi1 ) / step ) + 1 );
	for( float theta = phi1; theta <= phi2; theta += step )
	{
		pts.push_back( extractEllipsePointAtAngle( ellipse, theta ) );
	}
}

void ellipsePoint( const cctag::numerical::geometry::Ellipse& ellipse, float theta, Eigen::Vector3f& pt )
{
	const float x = ellipse.a() * cos( theta );
	const float y = ellipse.b() * sin( theta );

	pt( 0 ) = x * cos( ellipse.angle() ) - y* sin( ellipse.angle() ) + ellipse.center().x();
	pt( 1 ) = x * sin( ellipse.angle() ) + y* cos( ellipse.angle() ) + ellipse.center().y();
	pt( 2 ) = 1;
}

void computeIntermediatePoints(const Ellipse & ellipse, Point2d<Eigen::Vector3i> & pt11, Point2d<Eigen::Vector3i> & pt12, Point2d<Eigen::Vector3i> & pt21, Point2d<Eigen::Vector3i> & pt22){

	float a = -ellipse.b() * std::sin( ellipse.angle() ) - ellipse.b() * std::cos( ellipse.angle() );
	float b = -ellipse.a() * std::cos( ellipse.angle() ) + ellipse.a() * std::sin( ellipse.angle() );

	const float t11 = std::atan2( -a, b );
	const float t12 = t11 + boost::math::constants::pi<float>();

	a = -ellipse.b()* std::sin( ellipse.angle() ) + ellipse.b() * std::cos( ellipse.angle() );
	b = -ellipse.a()* std::cos( ellipse.angle() ) - ellipse.a() * std::sin( ellipse.angle() );

	const float t21 = std::atan2( -a, b );
	const float t22 = t21 + boost::math::constants::pi<float>();

	Eigen::Vector3f v11;
	ellipsePoint( ellipse, t11, v11 );
	Eigen::Vector3f v12;
	ellipsePoint( ellipse, t12, v12 );
	Eigen::Vector3f v21;
	ellipsePoint( ellipse, t21, v21 );
	Eigen::Vector3f v22;
	ellipsePoint( ellipse, t22, v22 );

	pt11.x() = (boost::math::round( v11( 0 ) ) );
	pt11.y() = (boost::math::round( v11( 1 ) ) );
	pt12.x() = (boost::math::round( v12( 0 ) ) );
	pt12.y() = (boost::math::round( v12( 1 ) ) );
	pt21.x() = (boost::math::round( v21( 0 ) ) );
	pt21.y() = (boost::math::round( v21( 1 ) ) );
	pt22.x() = (boost::math::round( v22( 0 ) ) );
	pt22.y() = (boost::math::round( v22( 1 ) ) );
}

void rasterizeEllipticalArc(const Ellipse & ellipse, const Point2d<Eigen::Vector3i> & pt1, const Point2d<Eigen::Vector3i> & pt2, std::vector< Point2d<Eigen::Vector3i> > & vPoint, std::size_t intersectionIndex)
{

	const int mx = std::abs(pt2.x() - pt1.x());
	const int my = std::abs(pt2.y() - pt1.y());

	if ( mx > my )
	{
		int x1 = std::min(pt1.x(),pt2.x());
		int x2 = std::max(pt1.x(),pt2.x());

		for( int x = x1 + 1 ; x < x2 ; ++x )
		{
			std::vector<float> intersections = intersectEllipseWithLine( ellipse, x, false );

			if( intersections.size() == 2 ){
				vPoint.emplace_back(x,boost::math::round(intersections[intersectionIndex]));
			}else if( intersections.size() == 1 ){
				vPoint.emplace_back(x,boost::math::round(intersections[0]));
			}
		}

	}else{
		int y1 = std::min(pt1.y(),pt2.y());
		int y2 = std::max(pt1.y(),pt2.y());

		for( int y = y1 + 1 ; y < y2 ; ++y )
		{
			std::vector<float> intersections = intersectEllipseWithLine( ellipse, y, true );

			if( intersections.size() == 2 ){
				vPoint.emplace_back(boost::math::round(intersections[intersectionIndex]), y);
			}else if( intersections.size() == 1 ){
				vPoint.emplace_back(boost::math::round(intersections[0]), y);
			}
		}
	}
}

/* Brief: Compute the intersection of an horizontal or a vertical line with an ellipse
 * Input:
 * ellipse
 * y: value/equation of the line
 * horizontal: if the line's equation is y=x then put true else vertical (false)
 * Return intersection(s) values.
 */

std::vector<float> intersectEllipseWithLine( const numerical::geometry::Ellipse& ellipse, float y, bool horizontal)
{
	using boost::math::pow;
	std::vector<float> res;

	float a, b, c;

	// y index of line, ordonate.
	const numerical::geometry::Ellipse::Matrix& eC = ellipse.matrix();

	if (horizontal){
		a = eC(0,0);
		b = 2 * ( y * eC(0,1) + eC(0,2) );
		c = eC(1,1) * pow<2>(y) + 2 * y * eC(2,1) + eC(2,2);
	}else{
		a = eC(1,1);
		b = 2 * ( y * eC(0,1) + eC(1,2) );
		c = eC(0,0) * pow<2>(y) + 2 * y * eC(0,2) + eC(2,2);
	}

	const float discriminant = pow<2>(b) / 4.0f - a * c;

	if( discriminant > 0.f )
	{
		// "General case" : 2 intersections x1, x2
		const float sqrtDiscriminant = std::sqrt( discriminant );
		res.push_back( ( -b / 2.0f - sqrtDiscriminant ) / a );
		res.push_back( ( -b / 2.0f + sqrtDiscriminant ) / a );
	}
	//@TODO make it more robust using fabs == epsilon
	else if( discriminant == 0.f )
	{
		// 1 point intersection
		res.push_back( -b / ( 2.0f * a ) );
	}
//	else if ( discriminant < 0 )
//	{
//		// No intersection
//	}
	return res;
}

void rasterizeEllipse( const Ellipse & ellipse, std::vector< Point2d<Eigen::Vector3i> > & vPoint )
{
	vPoint.reserve(int(ellipse.a()+ellipse.b())*2);

	Point2d<Eigen::Vector3i> pt11, pt12, pt21, pt22;
	computeIntermediatePoints(ellipse, pt11, pt12, pt21, pt22);

	Point2d<Eigen::Vector3i> ptAux;

	if ( pt11.x() > pt12.x() ){
		ptAux = pt11;
		pt11 = pt12;
		pt12 = ptAux;
	}

	if ( pt21.y() > pt22.y() ){
		ptAux = pt21;
		pt21 = pt22;
		pt22 = ptAux;
	}

	vPoint.push_back(pt11);
	vPoint.push_back(pt12);
	vPoint.push_back(pt21);
	vPoint.push_back(pt22);

	rasterizeEllipticalArc( ellipse, pt11, pt22, vPoint, 1);
	rasterizeEllipticalArc( ellipse, pt22, pt12, vPoint, 1);
	rasterizeEllipticalArc( ellipse, pt11, pt21, vPoint, 0);
	rasterizeEllipticalArc( ellipse, pt21, pt12, vPoint, 0);

	return;
}

std::size_t rasterizeEllipsePerimeter( const Ellipse & ellipse )
{
	Point2d<Eigen::Vector3i> pt11, pt12, pt21, pt22;
	
	computeIntermediatePoints(ellipse, pt11,pt12,pt21,pt22);

	// Determine diffx, diffy
	float diff1 = 0, diff2 = 0;

	{
		const float mx = std::abs( pt22.x() - pt11.x() );
		const float my = std::abs( pt22.y() - pt11.y() );
		if ( mx > my )
		{
			diff1 = mx;
		}
		else
		{
			diff1 = my;
		}
	}

	{
		const float mx = std::abs( pt12.x() - pt22.x() );
		const float my = std::abs( pt12.y() - pt22.y() );
		if ( mx > my )
		{
			diff2 = mx;
		}
		else
		{
			diff2 = my;
		}
	}

	return ( diff1 + diff2 ) * 2.f;
}


}
}
}


