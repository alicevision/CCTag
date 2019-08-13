/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_NOCUDA_RESIZE_HPP_
#define _CCTAG_NOCUDA_RESIZE_HPP_

#include "cctag/PlaneCV.hpp"

namespace cctag {

void resize( const Plane<uint8_t>& src, Plane<uint8_t>& dst );

}
#endif
