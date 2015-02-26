/*------------------------------------------------------------------------------

  Copyright (c) 2012-2014 viorica patraucean (vpatrauc@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

------------------------------------------------------------------------------*/
#include "EdgePoint.hpp"
#include <cctag/geometry/Ellipse.hpp>
#include <boost/foreach.hpp>

#ifndef _CCTAG_ELLIPSE_FIT_WITH_GRADIENTS_H
#define _CCTAG_ELLIPSE_FIT_WITH_GRADIENTS_H

namespace rom {
namespace vision {
namespace geometry {

void ellipse_fit_with_gradients( double *pts, double *grad, int pts_size,
                                 double **buff, int *size_buff_max,
                                 double *param );

void ellipseFittingWithGradientsToto( const std::vector<EdgePoint *> & vPoint, rom::numerical::geometry::Ellipse & ellipse );

}
}
}

#endif
