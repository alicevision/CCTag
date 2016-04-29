#ifndef _CCTAG_FITTING_H
#define _CCTAG_FITTING_H

#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Point.hpp>
#include <boost/foreach.hpp>

#include <list>
#include <string>
#include <vector>

namespace cctag {
namespace numerical {


// Precondition : pts.size >=2
// TODO déplacer innerProdMin
double innerProdMin( const std::vector<cctag::EdgePoint*>& childrens, double thrCosDiffMax, Point2d<Eigen::Vector3i> & p1, Point2d<Eigen::Vector3i> & p2 );

void circleFitting(cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& points);

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector< Point2d<Eigen::Vector3f> >& childrens );

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::vector<cctag::EdgePoint*>& childrens );

void ellipseFitting( cctag::numerical::geometry::Ellipse& e, const std::list<cctag::EdgePoint*>& childrens );

void ellipse_fit_with_gradients( double *pts, double *grad, int pts_size,
                                 double **buff, int *size_buff_max,
                                 double *param );

void ellipseFittingWithGradientsToto( const std::vector<EdgePoint *> & vPoint, cctag::numerical::geometry::Ellipse & ellipse );

bool matrixFromFile( const std::string& filename, std::list<cctag::EdgePoint>& edgepoints );

int discreteEllipsePerimeter( const cctag::numerical::geometry::Ellipse& ellipse);
} // namespace numerical
} // namespace cctag

#endif
