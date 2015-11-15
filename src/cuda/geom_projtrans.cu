#include "geom_projtrans.h"

namespace popart {
namespace geometry {

void projectiveTransform( const matrix3x3& transf, ellipse& e )
{
    matrix3x3_tView transf_transposed( trans );
    e.setMatrix(
        prod( transf_transposed,
              prod( e.matrix(),
                    transf ) ) );
}

}; // namespace geometry
}; // namespace popart

