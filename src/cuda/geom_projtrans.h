#pragma once

#include "geom_ellipse.h"

namespace popart {
namespace geometry {

void projectiveTransform( const matrix3x3& tr, ellipse& e );

}; // namespace geometry
}; // namespace popart

