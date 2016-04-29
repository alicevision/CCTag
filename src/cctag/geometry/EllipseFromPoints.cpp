#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/pow.hpp>

namespace cctag {
namespace numerical {
namespace geometry {

/**
 * @brief Extract one point from the ellipse at angle around the center (2D certer or image center ??).
 * @param[theta] angle of the computed point in radian
 * @todo lilian check this documentation !!
 */
Point2d<Eigen::Vector3f> extractEllipsePointAtAngle( const Ellipse & ellipse, double theta )
{
    Point2d<Eigen::Vector3f> p;
	theta = fmod( theta, 2 * boost::math::constants::pi<double>() );
    double x = ellipse.a() * cos( theta );
    double y = ellipse.b() * sin( theta );
    p.x() = ( x * cos( ellipse.angle() ) - y * sin( ellipse.angle() ) + ellipse.center().x() );
    p.y() = ( x * sin( ellipse.angle() ) + y * cos( ellipse.angle() ) + ellipse.center().y() );
    return p;
}

void points( const Ellipse & ellipse, const std::size_t nb, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts )
{
	const double step = 2.0 * boost::math::constants::pi<double>() / nb;
	points( ellipse, nb, step, 2 * boost::math::constants::pi<double>(), pts );
}

void points( const Ellipse & ellipse, const std::size_t nb, const double phi1, const double phi2, std::vector< cctag::Point2d<Eigen::Vector3f> > & pts )
{
	const double step = 2.0 * boost::math::constants::pi<double>() / nb;
	pts.reserve( std::size_t( ( phi2 - phi1 ) / step ) + 1 );
	for( double theta = phi1; theta <= phi2; theta += step )
	{
		pts.push_back( extractEllipsePointAtAngle( ellipse, theta ) );
	}
}

void ellipsePoint( const cctag::numerical::geometry::Ellipse& ellipse, double theta, Eigen::Vector3f& pt )
{
	const double x = ellipse.a() * cos( theta );
	const double y = ellipse.b() * sin( theta );

	pt( 0 ) = x * cos( ellipse.angle() ) - y* sin( ellipse.angle() ) + ellipse.center().x();
	pt( 1 ) = x * sin( ellipse.angle() ) + y* cos( ellipse.angle() ) + ellipse.center().y();
	pt( 2 ) = 1;
}

void computeIntermediatePoints(const Ellipse & ellipse, Point2d<Eigen::Vector3i> & pt11, Point2d<Eigen::Vector3i> & pt12, Point2d<Eigen::Vector3i> & pt21, Point2d<Eigen::Vector3i> & pt22){

	double a = -ellipse.b() * std::sin( ellipse.angle() ) - ellipse.b() * std::cos( ellipse.angle() );
	double b = -ellipse.a() * std::cos( ellipse.angle() ) + ellipse.a() * std::sin( ellipse.angle() );

	const double t11 = std::atan2( -a, b );
	const double t12 = t11 + boost::math::constants::pi<double>();

	a = -ellipse.b()* std::sin( ellipse.angle() ) + ellipse.b() * std::cos( ellipse.angle() );
	b = -ellipse.a()* std::cos( ellipse.angle() ) - ellipse.a() * std::sin( ellipse.angle() );

	const double t21 = std::atan2( -a, b );
	const double t22 = t21 + boost::math::constants::pi<double>();

	cctag::numerical::BoundedVector3d v11;
	ellipsePoint( ellipse, t11, v11 );
	cctag::numerical::BoundedVector3d v12;
	ellipsePoint( ellipse, t12, v12 );
	cctag::numerical::BoundedVector3d v21;
	ellipsePoint( ellipse, t21, v21 );
	cctag::numerical::BoundedVector3d v22;
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

void rasterizeEllipticalArc(const Ellipse & ellipse, const Point2d<Eigen::Vector3i> & pt1, const Point2d<Eigen::Vector3i> & pt2, std::vector< Point2d<Eigen::Vector3i> > & vPoint, std::size_t intersectionIndex){

	const double xCenter = ellipse.center().x();
	const double yCenter = ellipse.center().y();

	const int mx = std::abs(pt2.x() - pt1.x());
	const int my = std::abs(pt2.y() - pt1.y());

	if ( mx > my )
	{
		int x1 = std::min(pt1.x(),pt2.x());
		int x2 = std::max(pt1.x(),pt2.x());

		for( int x = x1 + 1 ; x < x2 ; ++x )
		{
			std::vector<double> intersections = intersectEllipseWithLine( ellipse, x, false );

			if( intersections.size() == 2 ){
				vPoint.push_back(Point2d<Eigen::Vector3i>(x,boost::math::round(intersections[intersectionIndex])));
			}else if( intersections.size() == 1 ){
				vPoint.push_back(Point2d<Eigen::Vector3i>(x,boost::math::round(intersections[0])));
			}
		}

	}else{
		int y1 = std::min(pt1.y(),pt2.y());
		int y2 = std::max(pt1.y(),pt2.y());

		for( int y = y1 + 1 ; y < y2 ; ++y )
		{
			std::vector<double> intersections = intersectEllipseWithLine( ellipse, y, true );

			if( intersections.size() == 2 ){
				vPoint.push_back(Point2d<Eigen::Vector3i>(boost::math::round(intersections[intersectionIndex]),y));
			}else if( intersections.size() == 1 ){
				vPoint.push_back(Point2d<Eigen::Vector3i>(boost::math::round(intersections[0]),y));
			}
		}
	}
}

/* Brief: Compute the intersection of an horizontal or vertical line with an ellipse
 * Input:
 * ellipse
 * y: value/equation of the line
 * horizontal: if the line's equation is y=x then put true else vertical (false)
 * Return intersection(s) values.
 */

std::vector<double> intersectEllipseWithLine( const numerical::geometry::Ellipse& ellipse, const double y, bool horizontal)
{
	using boost::math::pow;
	std::vector<double> res;

	double a, b, c;

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

	const double discriminant = pow<2>(b) / 4.0 - a * c;

	if( discriminant > 0 )
	{
		// "General case" : 2 intersections x1, x2
		const double sqrtDiscriminant = std::sqrt( discriminant );
		res.push_back( ( -b / 2.0 - sqrtDiscriminant ) / a );
		res.push_back( ( -b / 2.0 + sqrtDiscriminant ) / a );
	}
	else if( discriminant == 0 )
	{
		// 1 point intersection
		res.push_back( -b / ( 2.0 * a ) );
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
	double diff1 = 0, diff2 = 0;

	{
		const double mx = std::abs( pt22.x() - pt11.x() );
		const double my = std::abs( pt22.y() - pt11.y() );
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
		const double mx = std::abs( pt12.x() - pt22.x() );
		const double my = std::abs( pt12.y() - pt22.y() );
		if ( mx > my )
		{
			diff2 = mx;
		}
		else
		{
			diff2 = my;
		}
	}

	return ( diff1 + diff2 ) * 2.0;
}


}
}
}


