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
