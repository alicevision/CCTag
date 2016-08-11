/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#if 0

#include "geom_ellipse.h"

namespace popart {
namespace geometry {

void projectiveTransform( const matrix3x3&       m, ellipse& e );
void projectiveTransform( const matrix3x3_tView& m, ellipse& e );

}; // namespace geometry
}; // namespace popart

#endif
