// From http://ljk.imag.fr/membres/Pierre.Saramito/rheolef/source_html/ublas-invert_8h-source.html

#ifndef _CCTAG_NUMERICAL_INVERT_MATRIX_HPP_
#define _CCTAG_NUMERICAL_INVERT_MATRIX_HPP_
#if 0
//
// The following code inverts the matrix input using LU-decomposition
// with backsubstitution of unit vectors.
// Reference: Numerical Recipies in C, 2nd ed., by Press, Teukolsky, Vetterling & Flannery.
//
// http://www.crystalclearsoftware.com/cgi-bin/boost_wiki/wiki.pl?action=browse&diff=1&id=LU_Matrix_Inversion
// Hope someone finds this useful. Regards, Fredrik Orderud.
//
// Last edited September 4, 2007 5:23 am
//
#include <cctag/algebra/Determinant.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace cctag {
namespace numerical {

template<class Matrix>
bool invert_2x2( const Matrix& A, Matrix& result )
{
	float detM = cctag::numerical::det(A);//A( 0, 0 ) * A( 1, 1 ) - A( 0, 1 ) * A( 1, 0 );

	if( detM == 0 )
	{
		return false;
	}

	result( 0, 0 ) = A( 1, 1 );
	result( 0, 1 ) = -A( 0, 1 );
	result( 1, 0 ) = -A( 1, 0 );
	result( 1, 1 ) = A( 0, 0 );
	result         = ( 1.f / detM ) * result;
	return true;
}

template<class Matrix>
bool invert_3x3( const Matrix& A, Matrix& result )
{
	using namespace boost::numeric::ublas;
	typedef typename Matrix::value_type T;
        
	T determinant =  cctag::numerical::det(A);

	if( determinant == 0 )
	{
		return false;
	}

	result( 0, 0 ) = (   A( 1, 1 ) * A( 2, 2 ) - A( 1, 2 ) * A( 2, 1 ) ) / determinant;
	result( 1, 0 ) = ( -A( 1, 0 ) * A( 2, 2 ) + A( 2, 0 ) * A( 1, 2 ) ) / determinant;
	result( 2, 0 ) = (   A( 1, 0 ) * A( 2, 1 ) - A( 2, 0 ) * A( 1, 1 ) ) / determinant;
	result( 0, 1 ) = ( -A( 0, 1 ) * A( 2, 2 ) + A( 2, 1 ) * A( 0, 2 ) ) / determinant;
	result( 1, 1 ) = (   A( 0, 0 ) * A( 2, 2 ) - A( 2, 0 ) * A( 0, 2 ) ) / determinant;
	result( 2, 1 ) = ( -A( 0, 0 ) * A( 2, 1 ) + A( 2, 0 ) * A( 0, 1 ) ) / determinant;
	result( 0, 2 ) = (   A( 0, 1 ) * A( 1, 2 ) - A( 1, 1 ) * A( 0, 2 ) ) / determinant;
	result( 1, 2 ) = ( -A( 0, 0 ) * A( 1, 2 ) + A( 1, 0 ) * A( 0, 2 ) ) / determinant;
	result( 2, 2 ) = (   A( 0, 0 ) * A( 1, 1 ) - A( 1, 0 ) * A( 0, 1 ) ) / determinant;
	return true;
}

// Matrix inversion routine.
// Uses lu_factorize and lu_substitute in uBLAS to invert a matrix
template<class Matrix>
bool invert( const Matrix& input, Matrix& inverse )
{
	using namespace boost::numeric::ublas;
	using namespace boost::numeric;
	typedef typename Matrix::value_type T;
	typedef permutation_matrix<std::size_t> pmatrix;

	try
	{
		// create a working copy of the input
		Matrix A( input );
		// create a permutation matrix for the LU-factorization
		pmatrix pm( A.size1() );

		// perform LU-factorization
		int res = lu_factorize( A, pm );
		if( res != 0 )
		{
			return false;
		}

		// create identity matrix of "inverse"
		inverse.assign( identity_matrix<T>( A.size1() ) );

		// backsubstitute to get the inverse
		lu_substitute( A, pm, inverse );
	}
	catch( std::exception& )
	{
		return false;
	}
	return true;
}

/**
 * Generic matrix inverter' specialization for square matrix of size 3.
 */
template<typename T>
bool invert( const boost::numeric::ublas::bounded_matrix<T, 3, 3>& A,
             boost::numeric::ublas::bounded_matrix<T, 3, 3>& result )
{
	return invert_3x3( A, result );
}

/**
 * Generic matrix inverter' specialization for square matrix of size 2.
 */
template<typename T>
bool invert( const boost::numeric::ublas::bounded_matrix<T, 2, 2>& A,
             boost::numeric::ublas::bounded_matrix<T, 2, 2>& result )
{
	return invert_2x2( A, result );
}

template<class Matrix>
Matrix invert( const Matrix& m, bool& is_singular )
{
	Matrix inv_m( m.size1(), m.size2() );

	is_singular = invert( m, inv_m );
	return inv_m;
}

// http://archives.free.net.ph/message/20080909.064313.59c122c4.fr.html
template<class Matrix>
float determinant( boost::numeric::ublas::matrix_expression<Matrix> const& mat_r )
{
	using namespace boost::numeric;
	using namespace boost::numeric::ublas;

	float det = 1.f;
	Matrix mLu( mat_r() );
	permutation_matrix<std::size_t> pivots( mat_r().size1() );
	bool is_singular = lu_factorize( mLu, pivots );
	if( !is_singular )
	{
		for( std::size_t i = 0; i < pivots.size(); ++i )
		{
			if( pivots( i ) != i )
			{
				det *= -1.f;
			}
			det *= mLu( i, i );
		}
	}
	else
	{
		det = 0.f;
	}
	return det;
}

}
}
#endif
#endif
