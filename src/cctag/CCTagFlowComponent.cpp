/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/CCTagFlowComponent.hpp>
#include <cctag/utils/Defines.hpp>

#include <boost/foreach.hpp>

namespace cctag
{

CCTagFlowComponent::CCTagFlowComponent(
  const EdgePointCollection& edgeCollection,
  const std::vector<EdgePoint*> & outerEllipsePoints,
  const std::list<EdgePoint*> & childrens,
  const std::vector<EdgePoint*> & filteredChildrens,
  const cctag::numerical::geometry::Ellipse & outerEllipse,
  const std::list<EdgePoint*> & convexEdgeSegment,
  const EdgePoint & seed,
  std::size_t nCircles)
  : _outerEllipse(outerEllipse)
  , _seed(seed)
  , _nCircles(nCircles)
{

  _outerEllipsePoints.reserve(outerEllipsePoints.size());

  for(const EdgePoint * e : outerEllipsePoints)
  {
    _outerEllipsePoints.emplace_back(*e);
  }

  for(const EdgePoint * e : convexEdgeSegment)
  {
    _convexEdgeSegment.emplace_back(*e);
  }

  setFieldLines(childrens, edgeCollection);
  setFilteredFieldLines(filteredChildrens, edgeCollection);

}

void CCTagFlowComponent::setFilteredFieldLines(const std::vector<EdgePoint*> & filteredChildrens, const EdgePointCollection& edgeCollection)
{
  _filteredFieldLines.resize(filteredChildrens.size());

  std::size_t i = 0;

  for(auto p : filteredChildrens)
  {
    int dir = -1;
    std::vector<EdgePoint> & vE = _filteredFieldLines[i];

    vE.reserve(_nCircles);
    vE.emplace_back(*p);

    for (std::size_t j = 1; j < _nCircles; ++j)
    {
      if (dir == -1)
      {
        p = edgeCollection.before(p);
      }
      else
      {
        p = edgeCollection.after(p);
      }

      vE.emplace_back(*p);

      dir = -dir;
    }
    ++i;
  }
}

void CCTagFlowComponent::setFieldLines(const std::list<EdgePoint*> & childrens, const EdgePointCollection& edgeCollection)
{
  _fieldLines.resize(childrens.size());

  std::size_t i = 0;

  for (std::list<EdgePoint*>::const_iterator it = childrens.begin(); it != childrens.end(); ++it)
  {
    int dir = -1;
    EdgePoint* p = *it;
    assert( p );

    std::vector<EdgePoint> & vE = _fieldLines[i];

    vE.reserve(_nCircles);
    vE.emplace_back(*p);

    for (std::size_t j = 1; j < _nCircles; ++j)
    {
      if (dir == -1)
      {
        p = edgeCollection.before(p);
      }
      else
      {
        p = edgeCollection.after(p);
      }

      assert( p );
      vE.emplace_back(*p);

      dir = -dir;
    }
    ++i;
  }
}

} // namespace cctag
