
namespace rom {
namespace numerical {

/**
 * @brief Compute full SVD
 * @param[in] a input matrix
 * @param[out] u output matrix
 * @param[out] v output matrix
 * @param[out] s output diagonal matrix
 */
template<class MatA, class MatU, class MatV, class MatS>
void svd( const MatA & a, MatU & u, MatV & v, MatS & s)
{
	typedef typename MatA::value_type T;
	namespace lapack = boost::numeric::bindings::lapack;

	using namespace boost::numeric::ublas;

	size_t m = a.size1(), n = a.size2();
	size_t minmn = m < n ? m : n;
	size_t lw;

	// working A, U, V
	matrix<T, column_major> aw = a;
	matrix<T, column_major> uw( m, m );
	matrix<T, column_major> vw( n, n );
	boost::numeric::ublas::vector<T> sw( minmn );

	lw = lapack::gesvd_work( 'M', 'A', 'A', aw );

	std::vector<T> w( lw );
	lapack::gesvd( 'A', 'A', aw, sw, uw, vw, w );
	u = uw;
	v = trans( vw );
	s = diagonal_matrix<T, column_major >( sw.size(), sw.data() );
}

/**
 * Singular values only
 * @param[in] a input matrix
 * @param[out] s output diagonal matrix
 */
template<class MatA, class MatS>
void ssvd( const MatA & a, MatS & s)
{
	typedef typename MatA::value_type T;
	namespace lapack = boost::numeric::bindings::lapack;
	using namespace boost::numeric::ublas;

	matrix<T, column_major> u;
	matrix<T, column_major> v;

	// working A
	matrix<T, column_major> aw = a;

	size_t m = a.size1(), n = a.size2();
	size_t minmn = m < n ? m : n;

	boost::numeric::ublas::vector<T> sw;
	sw.resize( minmn );
	u.resize( m, m );
	v.resize( n, n );

	lapack::gesvd( 'M', 'N', 'N', aw, sw, u, v );
	s = MatS( sw.size(), sw.data() );
}

/**
 * @brief Compute partial SVD
 * @param[in] a input matrix
 * @param[out] u output matrix
 * @param[out] v output matrix
 * @param[out] s output diagonal matrix
 * @param[in] k output size
 */
template<class MatA, class MatU, class MatV, class MatS>
inline void svds( const MatA & a, MatU & u, MatV & v, MatS & s, const std::size_t k )
{
	typedef typename MatA::value_type T;
	using namespace boost::numeric::ublas;

	matrix<T, column_major> aw = a;
	matrix<T, column_major> uw;
	matrix<T, column_major> vw;
	diagonal_matrix<T, column_major> st;
	svd( aw, uw, vw, st );

	u = matrix_slice< matrix<T, column_major> > ( uw, slice( 0, 1, uw.size1() ), slice( 0, 1, k ) );
	v = matrix_slice< matrix<T, column_major> > ( vw, slice( 0, 1, vw.size1() ), slice( 0, 1, k ) );
	s = matrix_slice< diagonal_matrix<T, column_major> > ( st, slice( 0, 1, k ), slice( 0, 1, k ) );
}

}
}


