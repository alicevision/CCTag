#ifndef VISION_MARKER_CCTAG_FLOWCOMPONENT_HPP
#define	VISION_MARKER_CCTAG_FLOWCOMPONENT_HPP

#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <boost/foreach.hpp>

#include <vector>

namespace cctag
{
namespace vision
{
namespace marker
{

class CCTagFlowComponent
{
public:
  CCTagFlowComponent();

  CCTagFlowComponent(const std::vector<EdgePoint*> & outerEllipsePoints,
                     const std::list<EdgePoint*> & childrens,
                     const std::vector<EdgePoint*> & filteredChildrens,
                     const cctag::numerical::geometry::Ellipse & outerEllipse,
                     const std::list<EdgePoint*> & convexEdgeSegment,
                     const EdgePoint & seed,
                     std::size_t nCircles);

  virtual ~CCTagFlowComponent();

  void setFieldLines(const std::list<EdgePoint*> & childrens);
  void setFilteredFieldLines(const std::vector<EdgePoint*> & filteredChildrens);

  std::vector<EdgePoint> _outerEllipsePoints;
  cctag::numerical::geometry::Ellipse _outerEllipse;
  std::vector<std::vector<EdgePoint> > _fieldLines;
  std::vector<std::vector<EdgePoint> > _filteredFieldLines;
  std::list<EdgePoint> _convexEdgeSegment;
  EdgePoint _seed;
  std::size_t _nCircles;

};
}
}
}

#endif	/* CCTAGFLOWCOMPONENT_HPP */
