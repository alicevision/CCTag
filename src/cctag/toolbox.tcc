#include <opencv/cv.hpp>

/* 
 * File:   toolbox.tcc
 * Author: lcalvet
 *
 * Created on 4 août 2014, 21:41
 */

#ifndef _ROM_TOOLBOX_TCC
#define	_ROM_TOOLBOX_TCC

namespace rom {
namespace numerical {

/*template<typename Con, typename T, bool isPointer>
void ellipseFitting( rom::numerical::geometry::Ellipse& e, const Con & points )
{
	std::vector<cv::Point2f> pts;
	pts.reserve( points.size() );

	BOOST_FOREACH( const T & p, points )
	{
		if (isPointer){
			pts.push_back( cv::Point2f( p->x(), p->y() ) );
		}else{
			pts.push_back( cv::Point2f( p.x(), p.y() ) );
		}
	}

	cv::RotatedRect rR = cv::fitEllipse( cv::Mat( pts ) );

	float xC           = rR.center.x;
	float yC           = rR.center.y;

	float b = rR.size.height / 2.f;
	float a = rR.size.width / 2.f;

	double angle = rR.angle * boost::math::constants::pi<double>() / 180.0;

	if ( ( a == 0) || ( b == 0 ) ){
		ROM_THROW( exception::BadHandle() << exception::dev( "Degenerate ellipse after cv::fitEllipse => line or point." ) );
	}

	e.setParameters( Point2dN<double>( xC, yC ), a, b, angle );
}*/

}
}


#endif	/* TOOLBOX_TCC */

