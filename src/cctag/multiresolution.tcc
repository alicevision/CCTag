#include "detection.hpp"
#include "canny.hpp"

#include <cctag/image.hpp>

#include <map>

namespace rom {
namespace vision {
namespace marker {

inline void clearDetectedMarkers( const std::map<std::size_t, CCTag::List> & pyramidMarkers, const boost::gil::rgb32f_view_t & cannyRGB, const std::size_t curLevel )
{
	using namespace boost::gil;
	typedef rgb32f_pixel_t Pixel;
	Pixel pixelZero; terry::numeric::pixel_zeros_t<Pixel>()( pixelZero );
	typedef std::map<std::size_t, CCTag::List> LeveledMarkersT;

	BOOST_FOREACH( const LeveledMarkersT::const_iterator::value_type & v, pyramidMarkers )
	{
		const std::size_t level = v.first;
		const double factor = std::pow( 2.0, (double)(curLevel - level) );
		const CCTag::List & markers = v.second;
		BOOST_FOREACH( const CCTag & tag, markers )
		{
			BOOST_FOREACH( const rom::numerical::geometry::Ellipse & ellipse, tag.ellipses() )
			{
				rom::numerical::geometry::Ellipse ellipseScaled = ellipse;
				// Scale center
				Point2dN<double> c = ellipseScaled.center();
				c.setX( c.x() * factor );
				c.setY( c.y() * factor );
				ellipseScaled.setCenter( c );
				// Scale demi axes
				ellipseScaled.setA( ellipseScaled.a() * factor );
				ellipseScaled.setB( ellipseScaled.b() * factor );
				// Erase ellipses
				fillEllipse( cannyRGB, ellipseScaled, pixelZero );
			}
		}
	}
}

}
}
}

