/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Canny.hpp>

#include "utils/Defines.hpp"

namespace cctag
{

void edgesPointsFromCanny(
        EdgePointCollection& edgeCollection,
        Plane<uint8_t>& edges,
        Plane<int16_t>& dx,
        Plane<int16_t>& dy )
{
  std::size_t width  = edges.getCols();
  std::size_t height = edges.getRows();
  
  for( int y = 0 ; y < height ; ++y )
  {
    for( int x = 0 ; x < width ; ++x )
    {
      if ( edges.at(x,y) == 255 )
      {
        edgeCollection.add_point(x, y, dx.at(x,y), dy.at(x,y));
      }
    }
  }
}

} // namespace cctag


