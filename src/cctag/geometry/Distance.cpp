#include <immintrin.h>
#include "Distance.hpp"

namespace cctag {
namespace numerical {

// Compute (point-polar) distance between a point and an ellipse represented by its 3x3 matrix.
// TODO@lilian: f is always equal to 1, remove it
// NOTE: Q IS SYMMTERIC!; Eigen stores matrices column-wise by default.
float distancePointEllipseScalar(const Eigen::Vector3f& p, const Eigen::Matrix3f& Q, const float f )
{
  using Vector6f = Eigen::Matrix<float, 6, 1>;
  Vector6f aux( 6 );

  float x = p.x();
  float y = p.y();

  aux( 0 ) = x*x;
  aux( 1 ) = 2 * x * y;
  aux( 2 ) = 2* x;
  aux( 3 ) = y * y;
  aux( 4 ) = 2* y;
  aux( 5 ) = 1;

  float tmp1  = p( 0 ) * Q( 0, 0 ) + p( 1 ) * Q( 0, 1 ) + p( 2 ) * Q( 0, 2 );
  float tmp2  = p( 0 ) * Q( 0, 1 ) + p( 1 ) * Q( 1, 1 ) + p( 2 ) * Q( 1, 2 );
  float denom = tmp1 * tmp1 + tmp2 * tmp2;

  Vector6f qL(6);
  qL( 0 ) = Q( 0, 0 ) ; qL( 1 ) = Q( 0, 1 ) ; qL( 2 ) = Q( 0, 2 ) ;
  qL( 3 ) = Q( 1, 1 ) ; qL( 4 ) = Q( 1, 2 ) ;
  qL( 5 ) = Q( 2, 2 );
  return boost::math::pow<2>( aux.dot(qL) ) / denom;
}

// Q is symmetric matrix
static __m256 unpack_q(const Eigen::Vector3f& Q)
{
  static const __m256i index = _mm256_set_epi32(0, 0,  8,  5,  4,  2,  1,  0);
  static const __m256i mask =  _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
  return _mm256_mask_i32gather_ps(_mm256_setzero_ps(), Q.data(), _mm256_load_si256(&index),
    _mm256_castsi256_ps(_mm256_load_si256(&mask)), 4);
}

// Q are coefficients of the ellipse matrix: [ 0 | 0 | Q22 | Q12 | Q11 | Q02 | Q01 | Q00 ]
__m256 distancePointEllipseAVX2(__m256 px, __m256 py, __m256 pz, __m256 Q)
{
  
}

}
}

