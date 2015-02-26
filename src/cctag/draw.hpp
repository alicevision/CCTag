#ifndef _CCTAG_VISION_CCTAG_DRAW_HPP_
#define _CCTAG_VISION_CCTAG_DRAW_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/boostCv/cvImage.hpp>
#include <cctag/geometry/ellipseFromPoints.hpp>

#include <terry/numeric/init.hpp>

#include <boost/gil/image.hpp>
#include <boost/foreach.hpp>

#include <vector>


namespace rom {
namespace vision {
namespace marker {
namespace cctag {

template<class View>
static bool fillEllipseLine( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
{
	std::vector<double> intersections = rom::numerical::geometry::intersectEllipseWithLine( ellipse, y, true );
	BOOST_ASSERT( intersections.size() <= 2 );
	if( intersections.size() == 2 )
	{
		std::fill( image.at(intersections[0], y), image.at(intersections[1], y), color );
	}
	else if( intersections.size() == 1 )
	{
		*image.at(intersections[0], y) = color;
	}
	else //if( intersections.size() == 0 )
	{
		return false;
	}
	return true;
}


template<class View>
static bool drawEllipseLineIntersections( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
{
	std::vector<double> intersections = rom::numerical::geometry::intersectEllipseWithLine( ellipse, y, true );
	BOOST_ASSERT( intersections.size() <= 2 );
	if( intersections.size() == 2 )
	{
		if( intersections[0] >= 0 && intersections[0] < image.width() )
			*image.at(intersections[0], y) = color;
		if( intersections[1] >= 0 && intersections[1] < image.width() )
			*image.at(intersections[1], y) = color;
	}
	else if( intersections.size() == 1 )
	{
		if( intersections[0] >= 0 && intersections[0] < image.width() )
			*image.at(intersections[0], y) = color;
	}
	else //if( intersections.size() == 0 )
	{
		return false;
	}
	return true;
}

} // namespace cctag

            /**
             * @brief Fill image under an ellipse with \param color.
             *
             * @param[out] image output image we fill in \param color.
             * @param[in] ellipse define the region to fill.
             * @param[in] color color used to fill.
             */
template<class View>
inline void fillEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
{
	const double yCenter = ellipse.center().y();

	// visit the bottom part of the ellipse
	for( std::ssize_t y = yCenter; y < image.height(); ++y )
	{
		if( ! cctag::fillEllipseLine( image, ellipse, color, y ) )
			break;
	}
	// visit the upper part of the ellipse
	for( std::ssize_t y = yCenter; y > 0; --y )
	{
		if( ! cctag::fillEllipseLine( image, ellipse, color, y ) )
			break;
	}
}

template<class View>
inline void drawEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
{
	std::vector< Point2dN<int> > vPoint;
	rom::numerical::geometry::rasterizeEllipse( ellipse, vPoint );

	BOOST_FOREACH(Point2dN<int> p, vPoint){
		if( (p.x() >= 0) && (p.x() < image.width()) && (p.y() >= 0) && (p.y() < image.height()) ){
			*image.at(p.x(), p.y()) = color;
		}
	}

	/*const double yCenter = ellipse.center().y();

	int maxY = std::max(int(yCenter),0);
	int minY = std::min(int(yCenter),int(image.height())-1);

	// visit the bottom part of the ellipse
	for( std::ssize_t y = maxY; y < image.height(); ++y )
	{
		if( ! cctag::drawEllipseLineIntersections( image, ellipse, color, y ) )
			break;
	}
	// visit the upper part of the ellipse
	for( std::ssize_t y = minY; y > 0; --y )
	{
		if( ! cctag::drawEllipseLineIntersections( image, ellipse, color, y ) )
			break;
	}*/
}


template<class View>
void drawMarkerOnGilImage(View& image, const CCTag& marker, bool drawScaledMarker = true)
{
        using namespace terry::numeric;
        using namespace boost::gil;
        typedef typename View::value_type Pixel;
        typedef typename channel_type<View>::type Channel;
        Pixel pixelRed, pixelGreen, pixelBlue, pixelYellow, pixelMagenta, pixelCyan;


        pixel_zeros_t<Pixel>()(pixelRed);
        get_color(pixelRed, red_t()) = channel_traits<Channel>::max_value();
        pixel_zeros_t<Pixel>()(pixelGreen);
        get_color(pixelGreen, green_t()) = channel_traits<Channel>::max_value();
        pixel_zeros_t<Pixel>()(pixelBlue);
        get_color(pixelBlue, blue_t()) = channel_traits<Channel>::max_value();

        pixel_zeros_t<Pixel>()(pixelYellow);
        get_color(pixelYellow, red_t()) = channel_traits<Channel>::max_value();
        get_color(pixelYellow, green_t()) = channel_traits<Channel>::max_value();
        //get_color(pixelYellow, blue_t()) =
        pixel_zeros_t<Pixel>()(pixelMagenta);
        get_color(pixelMagenta, green_t()) = channel_traits<Channel>::max_value();
        //get_color(pixelMagenta, red_t()) =
        get_color(pixelMagenta, blue_t()) = channel_traits<Channel>::max_value();
        pixel_zeros_t<Pixel>()(pixelCyan);
        //get_color(pixelCyan, blue_t()) =
        get_color(pixelCyan, green_t()) = channel_traits<Channel>::max_value();
        get_color(pixelCyan, blue_t()) = channel_traits<Channel>::max_value();


        rom::numerical::geometry::Ellipse rescaledOuterEllipse;

        // Display ellipses
        if (drawScaledMarker) {
            rescaledOuterEllipse = marker.rescaledOuterEllipse();
        } else {
            rescaledOuterEllipse = marker.outerEllipse();
        }

        if (marker.getStatus() == rom::vision::marker::no_collected_cuts) {
            drawEllipse(image, rescaledOuterEllipse, pixelMagenta);
        }else if (marker.getStatus() == rom::vision::marker::no_selected_cuts) {
            drawEllipse(image, rescaledOuterEllipse, pixelCyan);
        }else if(marker.getStatus() == rom::vision::marker::opti_has_diverged){
            drawEllipse(image, rescaledOuterEllipse, pixelRed);
        }else if(marker.getStatus() == rom::vision::marker::id_not_reliable){
            drawEllipse(image, rescaledOuterEllipse, pixelCyan);
        }else if(marker.getStatus() == rom::vision::marker::id_reliable){
            drawEllipse(image, rescaledOuterEllipse, pixelGreen);
        }else if(marker.getStatus() == 0 ){
            drawEllipse(image, rescaledOuterEllipse, pixelGreen);
        }
            //BOOST_FOREACH(const rom::numerical::geometry::Ellipse & ellipse, marker.ellipses() )
            //{
            //	drawEllipse( image, ellipse, pixelGreen );
            //}

        ///@todo draw center image and id
}

// void drawMarkerInfos(SView & sourceView, const CCTag& marker, bool drawScaledMarker = true)
template<class View>
void drawMarkerInfos(View & sourceView, const CCTag& marker, bool drawScaledMarker = true)
{
        boostCv::CvImageView cvview(sourceView);
        IplImage * img = cvview.get();
        CvFont font1;
        cvInitFont(&font1, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 0, 2);

        std::string sId = boost::lexical_cast<std::string>(marker.id() + 1);

        int x, y;
        if (drawScaledMarker) {
            x = int (marker.rescaledOuterEllipse().center().x());
            y = int (marker.rescaledOuterEllipse().center().y());
        } else {
            x = int (marker.outerEllipse().center().x());
            y = int (marker.outerEllipse().center().y());
        }

        cvPutText(img, sId.c_str(),
                cvPoint(x, y),
                &font1, CV_RGB(128, 255, 0));
}

template<class View>
void drawMarkersOnGilImage(View& image, const CCTag::Vector& markers)
{
    BOOST_FOREACH(const CCTag& m, markers) {
        drawMarkerOnGilImage(image, m);
        drawMarkerInfos(image, m);
    }
}



} // namespace marker
} // namespace vision
} // namespace rom

#endif
