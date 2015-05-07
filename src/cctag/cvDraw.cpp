#include "cvDraw.hpp"

#include <opencv2/core/core_c.h>


namespace cctag {

void drawMarkerOnImage( IplImage* simg, const CCTag& marker )
{
	// Display ellipses
	const cctag::numerical::geometry::Ellipse & e = marker.outerEllipse();
	const double theta = e.angle() * 180.0 / boost::math::constants::pi<double>();
	if ( marker.id() != 0 )
	{
		cvEllipse( simg, cvPoint( (int)boost::math::round( e.center().x() ), (int)boost::math::round( e.center().y() ) ), cvSize( (int)e.a(), (int)e.b() ), theta, 0, 360, CV_RGB( 0, 255, 0 ), 1, 8, 0 );
	}
	else
	{
		cvEllipse( simg, cvPoint( (int)boost::math::round( e.center().x() ), (int)boost::math::round( e.center().y() ) ), cvSize( (int)e.a(), (int)e.b() ), theta, 0, 360, CV_RGB( 0, 0, 255 ), 1, 8, 0 );
	}
	
	///@todo draw center image and id
	
	CCTAG_COUT_LILIAN( e.matrix() );
}

void drawMarkersOnImage( IplImage* simg, const CCTag::Vector& markers )
{
	BOOST_FOREACH( const CCTag& m, markers )
	{
		CCTAG_COUT_LILIAN( m.ellipses()[m.ellipses().size()-1].matrix() );

		CCTAG_COUT_LILIAN( "Radius ratio : " );

		for( std::size_t n = 0 ; n < m.radiusRatios().size() ; n++ )
		{
			CCTAG_COUT_LILIAN( m.radiusRatios()[n] );
		}

//			std::cout << "Id : " << m.id() << std::endl;
//			if (m.id() != -1)
//				stream << m << std::endl;

		drawMarkerOnImage( simg, m );
	}
}

} // namespace cctag
