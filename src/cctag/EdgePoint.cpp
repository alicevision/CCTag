/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/EdgePoint.hpp>

namespace cctag
{

std::ostream& operator<<( std::ostream& os, const EdgePoint& eP )
{
  os << "quiver( " << eP.x() << " , " << eP.y() << "," << eP.dX() << "," << eP.dY() << " ); ";
  return os;
}

} // namespace cctag
