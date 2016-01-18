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
