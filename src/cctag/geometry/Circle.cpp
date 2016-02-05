#include <cctag/Global.hpp>
#include <cctag/geometry/Circle.hpp>
#include <cctag/geometry/Point.hpp>


namespace cctag {
namespace numerical {
namespace geometry {

Circle::Circle( const Point2dN<double>& center, const double r )
	: Ellipse( center, r, r, 0.0 )
{
}

Circle::Circle( const double r )
	: Ellipse( Point2dN<double>(0.0, 0.0) , r, r, 0.0 )
{
}

Circle::~Circle()
{}

}
}
}
