#ifndef _CCTAG_VISION_MARKER_TOOLBOX_HPP_
#define _CCTAG_VISION_MARKER_TOOLBOX_HPP_

#include <list>
#include <string>
#include <vector>

#include <cctag/geometry/point.hpp>

namespace popart {
namespace vision {
class EdgePoint;
}
namespace numerical {
namespace geometry {
class Ellipse;
}
}
}

namespace popart {
namespace numerical {

// Precondition : pts.size >=2
// TODO déplacer innerProdMin
double innerProdMin( const std::vector<popart::vision::EdgePoint*>& childrens, double thrCosDiffMax, Point2dN<int> & p1, Point2dN<int> & p2 );

void circleFitting(popart::numerical::geometry::Ellipse& e, const std::vector<popart::vision::EdgePoint*>& points);

void ellipseFitting( popart::numerical::geometry::Ellipse& e, const std::vector< Point2dN<double> >& childrens );

void ellipseFitting( popart::numerical::geometry::Ellipse& e, const std::vector<popart::vision::EdgePoint*>& childrens );

void ellipseFitting( popart::numerical::geometry::Ellipse& e, const std::list<popart::vision::EdgePoint*>& childrens );

bool matrixFromFile( const std::string& filename, std::list<popart::vision::EdgePoint>& edgepoints );

int discreteEllipsePerimeter( const popart::numerical::geometry::Ellipse& ellipse);

}
}

#endif
