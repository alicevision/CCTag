#include <cctag/utils/Defines.hpp>
#include <cctag/geometry/Circle.hpp>
#include <cctag/geometry/Point.hpp>


namespace cctag {
namespace numerical {
namespace geometry {

Circle::Circle( const Point2d<Eigen::Vector3f>& center, const float r )
	: Ellipse( center, r, r, 0.f )
{
}

Circle::Circle( const float r )
	: Ellipse( Point2d<Eigen::Vector3f>(0.f, 0.f) , r, r, 0.f )
{
}

}
}
}
