
namespace rom {
namespace vision {
namespace marker {
namespace cctag {

template<class View>
inline bool fillEllipseLine( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
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
inline bool drawEllipseLineIntersections( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color, const std::size_t y )
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

}

template<class View>
void fillEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
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
void drawEllipse( View& image, const numerical::geometry::Ellipse& ellipse, const typename View::value_type& color )
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


}
}
}

