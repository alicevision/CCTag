#ifndef _CCTAG_NUMERICAL_DETERMINANT_HPP_
#define	_CCTAG_NUMERICAL_DETERMINANT_HPP_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace popart {
namespace numerical {

namespace ublas = boost::numeric::ublas;

template<typename T>
T det(const ublas::bounded_matrix<T,2,2> & m)
{
    T det =  m( 0, 0 ) * m( 1, 1 ) - m( 0, 1 ) * m( 1, 0 );

    return det;
}


template<typename T>
T det(const ublas::bounded_matrix<T,3,3> & m)
{
    T det =  m( 0, 0 ) * ( m( 1, 1 ) * m( 2, 2 ) - m( 2, 1 ) * m( 1, 2 ) )
                       - m( 0, 1 ) * ( m( 1, 0 ) * m( 2, 2 ) - m( 1, 2 ) * m( 2, 0 ) )
                       + m( 0, 2 ) * ( m( 1, 0 ) * m( 2, 1 ) - m( 1, 1 ) * m( 2, 0 ) ) ;

    return det;
}

namespace detail
{
	inline int determinant_sign(const ublas::permutation_matrix<std::size_t>& pm)
	{
		int pm_sign=1;
		std::size_t size = pm.size();
		for ( std::size_t i = 0; i < size; ++i )
		{
			if (i != pm(i))
			{
				pm_sign *= -1; // swap_rows would swap a pair of rows here, so we change sign
			}
		}
		return pm_sign;
	}
}

template<class MatA>
typename MatA::value_type determinant( const MatA & a )
{
	typedef typename MatA::value_type T;
	ublas::matrix<T> m = a;
    ublas::permutation_matrix<std::size_t> pm( m.size1() );
    T det = 1;
    if( ublas::lu_factorize( m, pm ) ) {
        det = 0;
    } else {
        for(int i = 0; i < m.size1(); i++)
		{
            det *= m(i,i); // multiply by elements on diagonal
		}
        det = det * detail::determinant_sign( pm );
    }
    return det;
}



}
}

#endif

