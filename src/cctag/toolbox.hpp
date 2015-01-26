#ifndef _ROM_VISION_MARKER_TOOLBOX_HPP_
#define _ROM_VISION_MARKER_TOOLBOX_HPP_

#include <list>
#include <string>
#include <vector>

#include <cctag/geometry/point.hpp>

namespace rom {
namespace vision {
class EdgePoint;
}
namespace numerical {
namespace geometry {
class Ellipse;
}
}
}

//#include "toolbox.tcc"

namespace rom {
namespace numerical {

// Precondition : pts.size >=2
// TODO déplacer innerProdMin
double innerProdMin( const std::vector<rom::vision::EdgePoint*>& childrens, double thrCosDiffMax, Point2dN<int> & p1, Point2dN<int> & p2 );

void circleFitting(rom::numerical::geometry::Ellipse& e, const std::vector<rom::vision::EdgePoint*>& points);

void ellipseFitting( rom::numerical::geometry::Ellipse& e, const std::vector< Point2dN<double> >& childrens );

void ellipseFitting( rom::numerical::geometry::Ellipse& e, const std::vector<rom::vision::EdgePoint*>& childrens );

void ellipseFitting( rom::numerical::geometry::Ellipse& e, const std::list<rom::vision::EdgePoint*>& childrens );

bool matrixFromFile( const std::string& filename, std::list<rom::vision::EdgePoint>& edgepoints );

int discreteEllipsePerimeter( const rom::numerical::geometry::Ellipse& ellipse);

}
}

#endif
