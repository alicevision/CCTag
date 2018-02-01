/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_CCTAG_FLOWCOMPONENT_HPP
#define	VISION_MARKER_CCTAG_FLOWCOMPONENT_HPP

#include <cctag/EdgePoint.hpp>
#include <cctag/geometry/Ellipse.hpp>

#include <vector>
#include <list>

#include "Types.hpp"

namespace cctag
{

class CCTagFlowComponent
{
public:
  CCTagFlowComponent()
  {}

  CCTagFlowComponent(const EdgePointCollection& edgeCollection,
                     const std::vector<EdgePoint*> & outerEllipsePoints,
                     const std::list<EdgePoint*> & children,
                     const std::vector<EdgePoint*> & filteredChildren,
                     const cctag::numerical::geometry::Ellipse & outerEllipse,
                     const std::list<EdgePoint*> & convexEdgeSegment,
                     const EdgePoint & seed,
                     std::size_t nCircles);

  void setFieldLines(const std::list<EdgePoint*> & children, const EdgePointCollection& edgeCollection);
  void setFilteredFieldLines(const std::vector<EdgePoint*> & filteredChildren, const EdgePointCollection& edgeCollection);

  std::vector<EdgePoint> _outerEllipsePoints;
  cctag::numerical::geometry::Ellipse _outerEllipse;
  std::vector<std::vector<EdgePoint> > _fieldLines;
  std::vector<std::vector<EdgePoint> > _filteredFieldLines;
  std::list<EdgePoint> _convexEdgeSegment;  // inner arc
  EdgePoint _seed;
  std::size_t _nCircles;

};

} // namespace cctag

#endif	/* CCTAGFLOWCOMPONENT_HPP */
