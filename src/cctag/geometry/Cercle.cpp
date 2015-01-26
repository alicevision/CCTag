#include "Cercle.hpp"

#include <cctag/geometry/point.hpp>
#include <cctag/global.hpp>

namespace rom {
namespace numerical {
namespace geometry {

Cercle::Cercle( const Point2dN<double>& center, const double r )
	: Ellipse( center, r, r, 0.0 )
{
}

Cercle::Cercle( const double r )
	: Ellipse( Point2dN<double>(0.0, 0.0) , r, r, 0.0 )
{
}

Cercle::~Cercle()
{}

}
}
}
