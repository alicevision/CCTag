#ifndef _CCTAG_EIG_HPP
#define	_CCTAG_EIG_HPP

#include <cctag/progBase/exceptions.hpp>

#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#include <boostLapackExtension/ggev.hpp>

namespace cctag {
namespace numerical {

	template<class MatA, class MatV, class MatD>
	inline void eigzl( const MatA & a, MatV & v, MatD & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		namespace ublas = boost::numeric::ublas;
		using namespace ublas;
		matrix<T, column_major> wa = a;
		matrix<T, column_major> wv( a.size1(), a.size2() );
		ublas::vector< T > dc( a.size1() );
		v.resize( a.size1(), a.size2() );
		v.resize( a.size1(), a.size2() );

		int ierr = lapack::geev( wa, dc, &wv, (matrix<T,column_major>*)(NULL), lapack::optimal_workspace() );

		if ( ierr < 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("eigval param error!") );
		}
		else if ( ierr > 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("no convergence for eigval!") );
		}

		v = wv;
		d.resize( a.size1(), a.size1() );
		for( std::size_t k = 0; k < dc.size(); ++k )
		{
			d( k, k ) = dc( k );
		}
	}

	template<class MatA, class MatV, class MatD>
	inline void eigzr( const MatA & a, MatV & v, MatD & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		namespace ublas = boost::numeric::ublas;

		ublas::matrix<T, ublas::column_major> wa = a;
		ublas::matrix<T, ublas::column_major> wv( a.size1(), a.size2() );
		ublas::vector< T > dc( a.size1() );
		v.resize( a.size1(), a.size2() );
		v.resize( a.size1(), a.size2() );

		int ierr = lapack::geev( wa, dc, (ublas::matrix<T,ublas::column_major>*)(NULL), &wv, lapack::optimal_workspace() );

		if ( ierr < 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("eigval param error!") );
		}
		else if ( ierr > 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("no convergence for eigval!") );
		}

		v = wv;
		d.resize( a.size1(), a.size1() );
		for( std::size_t k = 0; k < dc.size(); ++k )
		{
			d( k, k ) = dc( k );
		}
	}

	template<class MatA, class MatV, class MatD>
	inline void eig( const MatA & a, MatV & v, MatD & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		namespace ublas = boost::numeric::ublas;
		using namespace ublas;

		matrix<T, column_major> wa = a;
		matrix<T, column_major> wv( a.size1(), a.size2() );
		ublas::vector< std::complex<T> > dc( a.size1() );
		v.resize( a.size1(), a.size2() );

		int ierr = lapack::geev( wa, dc, &wv, (matrix<T,column_major>*)(NULL), lapack::optimal_workspace() );

		if ( ierr < 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("eigval param error!") );
		}
		else if ( ierr > 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("no convergence for eigval!") );
		}

		v = wv;
		d.resize( a.size1(), a.size1() );
		for( std::size_t k = 0; k < d.size1(); ++k )
		{
			d( k, k ) = dc( k ).real();
		}
	}

	template<class MatA, class MatB, class MatC, class MatD>
	inline void eig( const MatA & a, const MatB & b, MatC & v, MatD & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		namespace ublas = boost::numeric::ublas;

		ublas::matrix< T, ublas::column_major > wa = a;
		ublas::matrix< T, ublas::column_major > wb = b;
		ublas::matrix< T, ublas::column_major > wvl( a.size1(), a.size2() );
		ublas::matrix< T, ublas::column_major> wvr( a.size1(), a.size2() );
		ublas::vector< T > alphar( a.size1() );
		ublas::vector< T > alphai( a.size1() );
		ublas::vector< T > beta( a.size1() );


		int ierr = lapack::ggev( 'N', 'V', wa, wb, wvl, wvr, alphar, alphai, beta );

		if ( ierr < 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("eigval param error!") );
		}
		else if ( ierr > 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("no convergence for eigval!") );
		}

		v = wvr;
		d.resize( a.size1(), a.size1() );
		for( std::size_t k = 0; k < a.size1(); ++k )
		{
			d( k, k ) = alphar( k ) / beta( k );
		}
	}

	template<class MatA, class MatD>
	inline void eigd( const MatA & a, MatD & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		namespace ublas = boost::numeric::ublas;

		ublas::matrix<T, ublas::column_major> wa = a;
		ublas::vector< std::complex<T> > dc( a.size1() );

		ublas::matrix<T, boost::numeric::ublas::column_major> vl( a.size1(), a.size2() ), vr( a.size1(), a.size2() );

		int ierr = lapack::geev( wa, dc, &vl, &vr, lapack::optimal_workspace() );
		if ( ierr < 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("eigval param error!") );
		}
		else if ( ierr > 0 )
		{
			ROM_THROW( exception::Bug()
				<< exception::dev("no convergence for eigval!") );
		}
		d.resize( a.size1(), a.size1() );
		for( std::size_t k = 0; k < a.size1(); ++k )
		{
			d( k, k ) = dc( k ).real();
		}
	}

	template<class MatA>
	inline void eigd( const MatA & a,
			          boost::numeric::ublas::vector<typename MatA::value_type> & d )
	{
		typedef typename MatA::value_type T;
		namespace lapack = boost::numeric::bindings::lapack;
		using namespace boost::numeric::ublas;

		diagonal_matrix<T, column_major> md;
		eigd( a, md );
		d.resize( a.size1() );
		for( std::size_t k = 0; k < a.size1(); ++k )
		{
			d( k ) = md( k, k );
		}
	}

}
}

#endif

