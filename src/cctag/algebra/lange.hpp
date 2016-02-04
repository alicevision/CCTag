#ifndef _LANGE_HPP
#define	_LANGE_HPP

#include "lapack_add.h"
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


namespace boost { namespace numeric { namespace bindings {

	namespace lapack {

		namespace detail {
			inline double lange( char norm, int m, int n, double *a, int lda, double *work )
			{
				return LAPACK_DLANGE( &norm, &m, &n, a, &lda, work );
			}
		}

		template <typename MatA >
		typename MatA::value_type lange ( char norm, MatA & a ) {
			typedef typename MatA::value_type T;

			if ( norm == 'I' || norm == 'i' )
			{
			    traits::detail::array<T> w( 2 * traits::matrix_size1 (a) );
				return detail::lange( norm,
									  traits::matrix_size1 (a),
									  traits::matrix_size2 (a),
									  traits::matrix_storage (a),
									  traits::leading_dimension (a),
									  traits::vector_storage (w) );
			}
			else
			{
				return detail::lange( norm,
									  traits::matrix_size1 (a),
									  traits::matrix_size2 (a),
									  traits::matrix_storage (a),
									  traits::leading_dimension (a),
									  NULL );
			}

		}
	}
}
}
}



#endif
