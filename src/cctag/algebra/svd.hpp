#ifndef _CCTAG_NUMERICAL_SVD_HPP_
#define	_CCTAG_NUMERICAL_SVD_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/matrix_traits.hpp>
#include <boost/numeric/bindings/lapack/lapack.hpp>

namespace popart {
namespace numerical {

/**
 * @brief Compute full SVD
 * @param[in] a input matrix
 * @param[out] u output matrix
 * @param[out] v output matrix
 * @param[out] s output diagonal matrix
 */
template<class MatA, class MatU, class MatV, class MatS>
void svd( const MatA & a, MatU & u, MatV & v, MatS & s);

/**
 * Singular values only
 * @param[in] a input matrix
 * @param[out] s output diagonal matrix
 */
template<class MatA, class MatS>
void ssvd( const MatA & a, MatS & s);

/**
 * @brief Compute partial SVD
 * @param[in] a input matrix
 * @param[out] u output matrix
 * @param[out] v output matrix
 * @param[out] s output diagonal matrix
 * @param[in] k output size
 */
template<class MatA, class MatU, class MatV, class MatS>
void svds( const MatA & a, MatU & u, MatV & v, MatS & s, const std::size_t k );

}
}

#include "svd.tcc"

#endif
