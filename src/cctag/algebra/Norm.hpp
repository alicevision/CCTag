#ifndef _CCTAG_NUMERICAL_NORM_HPP_
#define	_CCTAG_NUMERICAL_NORM_HPP_

#include <cctag/algebra/Svd.hpp>

namespace cctag {
namespace numerical {


	/**
	 *@return The largest singular value ( max( svd( a ) ) )
	 */
	template<class MatA>
	inline typename MatA::value_type norm2( const MatA & a )
	{
		typedef typename MatA::value_type T;
		using namespace boost::numeric::ublas;
		diagonal_matrix<T, column_major> s;
		ssvd( a, s );
		T maxs = 0;
		if ( s.size1() > 0 && s.size2() > 0 )
		{
			maxs = s( 0, 0 );
			for( std::size_t k = 0; k < s.size1(); ++k )
			{
				T val = s( k, k );
				if ( maxs < val )
				{
					maxs = val;
				}
			}
		}
		return maxs;
	}
}
}


#endif
