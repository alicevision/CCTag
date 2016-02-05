#ifndef _GGEV_HPP
#define	_GGEV_HPP
#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/workspace.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/ublas/io.hpp>

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
#  include <boost/static_assert.hpp>
#  include <boost/type_traits.hpp>
#endif

#include <cassert>

#ifdef	__cplusplus
extern "C" {
#endif

/********************************************/
/* generalized eigenvalue/eigenvector */


	void LAPACK_DGGEV(char const* jobz, char *jobvr, int const* n,
                     double* a, int const* lda, double* b, int const* ldb,
                     double *alphar, double *alphai,
					 double *beta, double *vl, int *ldvl, double *vr,
					 int *ldvr, double *work, int *lwork, int *info);

/********************************************/
/* norm computation */

	double LAPACK_DLANGE( char *norm, int *m, int *n, double *a, int *lda, double *work );

#ifdef	__cplusplus
}
#endif

namespace boost { namespace numeric { namespace bindings {

  namespace lapack {

    namespace detail {

      inline
      void ggev ( char jobz, char jobvr, int const n,
                  double* a, int const lda, double* b, int const ldb,
                  double *alphar, double *alphai,
				  double *beta, double *vl, int ldvl, double *vr,
				  int ldvr, double *work, int lwork, int &info )
      {
        LAPACK_DGGEV (&jobz, &jobvr, &n, a, &lda, b, &ldb, alphar, alphai, beta, vl, &ldvl, vr, &ldvr, work, &lwork, &info );
      }
	}

    template <typename A, typename B, typename V, typename AR>
    int ggev (char jobz, char jobvr, A& a, B& b, V& vl, V& vr, AR& alphar, AR& alphai, AR& beta) {

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<A>::matrix_structure,
        traits::general_t
      >::value));
#endif
	  
#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
      typedef typename traits::matrix_traits<A>::value_type val_t;
#else
      typedef typename MatrA::value_type val_t;
#endif
      typedef typename traits::type_traits<val_t>::real_type real_t;

      int const n = traits::matrix_size1 (a);
      assert ( n>0 );
      assert (traits::matrix_size2 (a)==n);
      assert (traits::leading_dimension (a)>=n);

      int const nb = traits::matrix_size1 (b);
      assert ( nb>0 );
      assert (traits::matrix_size2 (b)==nb);
      assert (traits::leading_dimension (b)>=nb);
      assert ( n== nb);

      assert ( jobz=='N' || jobz=='V' );
      assert ( jobvr=='N' || jobvr=='V' );
	  traits::detail::array<real_t> w( 16 * n );

      int info;
      detail::ggev(
                   jobz, jobvr, n,
                   traits::matrix_storage (a),
                   traits::leading_dimension (a),
                   traits::matrix_storage (b),
                   traits::leading_dimension (b),
                   traits::vector_storage (alphar),
                   traits::vector_storage (alphai),
                   traits::vector_storage (beta),
                   traits::matrix_storage (vl),
                   traits::leading_dimension (vl),
                   traits::matrix_storage (vr),
                   traits::leading_dimension (vr),
                   traits::vector_storage (w),
                   traits::vector_size (w),
                   info);
      return info;
    }
  }
}
}
}

#endif	/* _GGEV_HPP */

