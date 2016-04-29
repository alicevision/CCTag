#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/Circle.hpp>
#include <cctag/geometry/Point.hpp>


namespace cctag {
namespace numerical {
namespace geometry {

Circle::Circle( const Point2d<Eigen::Vector3f>& center, const double r )
	: Ellipse( center, r, r, 0.0 )
{
}

Circle::Circle( const double r )
	: Ellipse( Point2d<Eigen::Vector3f>(0.0, 0.0) , r, r, 0.0 )
{
}

}
}
}
