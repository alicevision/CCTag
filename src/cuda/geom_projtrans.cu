/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#if 0
#include "geom_projtrans.h"

namespace popart {
namespace geometry {

void projectiveTransform( const matrix3x3& m, ellipse& e )
{
    matrix3x3_tView m_transposed( m );
    e.setMatrix(
        prod( m_transposed,
              prod( e.matrix(),
                    m ) ) );
}

void projectiveTransform( const matrix3x3_tView& m, ellipse& e )
{
    e.setMatrix(
        prod( m,
              prod( e.matrix(),
                    m.transposed() ) ) );
}

}; // namespace geometry
}; // namespace popart

#endif
