#ifndef _CCTAG_HOMOGRAPHYCOMPUTATION_HPP
#define	_CCTAG_HOMOGRAPHYCOMPUTATION_HPP

#include "CCTag.hpp"
#include <cctag/geometry/Ellipse.hpp>

#include <cctag/algebra/eig.hpp>
#include <cctag/algebra/lapack.hpp>
#include <cctag/algebra/svd.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/optimization/conditioner.hpp>

#include <boost/foreach.hpp>
//#include <boost/numeric/ublas/banded.hpp>
//#include <boost/numeric/ublas/vector.hpp>
#include "boost/numeric/ublas/fwd.hpp"

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace rom {
namespace vision {
namespace marker {

void conditionnate( std::pair<rom::vision::marker::CCTag, rom::vision::marker::CCTag> & frameMarkers, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mT, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & mInvT );

/**
  V = M * diag( 1./ sqrt(sum(M.*M)));
 */
template<class M>
inline boost::numeric::ublas::vector<typename M::value_type> unit( const M & m )
{
	namespace ublas = boost::numeric::ublas;

	typedef typename M::value_type T;
	ublas::vector<T> vec( m.size1() );
	// Square sum
	T sumsq = 0;
	for( std::size_t i = 0; i < m.size1(); ++i )
	{
		for( std::size_t j = 0; j < m.size2(); ++j )
		{
			sumsq += m( i, j ) * m( i, j );
		}
	}
	T val = 1.0 / std::sqrt( sumsq );
	for( std::size_t k = 0; k < m.size1(); ++k )
	{
		vec( k ) = val;
	}

	return prod( m, vec );
}

/**
function Z = null(A)
   [m,n] = size(A);
   % Orthonormal basis

   [U,S,V] = svd(A,0);
   if m > 1, s = diag(S);
      elseif m == 1, s = S(1);
      else s = 0;
   end
   tol = max(m,n) * max(s) * eps(class(A)); % voir help eps matlab (eps('double') = 2.220446049250313e-16)
   r = sum(s > tol);
   Z = V(:,r+1:n);
return;
*/
template<class T>
boost::numeric::ublas::matrix<T> null( const boost::numeric::ublas::matrix<T> & a )
{
	namespace ublas = boost::numeric::ublas;

	size_t m = a.size1(), n = a.size2();
	ublas::matrix<T> u,v,z;
	ublas::matrix<T> s;
	rom::numerical::svd( a, u, v, s );
	ublas::vector<T> sd( s.size1() );
	T smax = 0;
	if ( s.size1() > 0 && s.size2() > 0 )
	{
		smax = s( 0, 0 );
		for( std::size_t k = 0; k < s.size1(); ++k )
		{
			T val = s( k, k );
			sd( k ) = val;
			if ( smax < val )
			{
				smax = val;
			}
		}
	}
	T tol = std::max( m, n ) * smax * std::numeric_limits<T>::epsilon();
	T r = 0;
	for( std::size_t k = 0; k < sd.size(); ++k )
	{
		if ( sd( k ) > tol )
		{
			r += sd( k );
		}
	}
	return ublas::matrix_range< ublas::matrix<T> > ( v, ublas::range( 0, v.size1() ), ublas::range( int ( std::ceil( r ) ), v.size2() ) );
}

/**
function L = linesFromRank2Conic( C )
[U,S,V]  = svds(C,2);
X        = U * sqrt(abs(S));
L(:,1)   = X(:,1)+X(:,2);
L(:,2)   = X(:,1)-X(:,2);
return;
*/
template<class MatC>
boost::numeric::ublas::matrix<typename MatC::value_type> linesFromRank2Conic( const MatC & c )
{
	typedef typename MatC::value_type T;
	using namespace boost::numeric::ublas;

	matrix<T> u,v,x,l;

	diagonal_matrix<T> s;
	rom::numerical::svds( c, u, v, s, 2 );
	for( std::size_t k = 0; k < s.size1(); ++k )
	{
		s( k, k ) = std::sqrt( std::abs( s( k, k ) ) );
	}
	x = prod( u, s );
	l.resize( x.size1(), 2 );
	for( std::size_t k = 0; k < x.size1(); ++k )
	{
		l( k, 0 ) = x( k, 0 ) + x( k, 1 );
		l( k, 1 ) = x( k, 0 ) - x( k, 1 );
	}
	return l;
}

/**
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% --------------------------------------------------------------------
function [s1,s2] = signature(X,epsil)
% --------------------------------------------------------------------
     if nargin==1, epsil=10e-8;  end
     X        = X/norm(X,2); % norme 2 de X, returns the largest singular value of A, max(svd(A))
     dg       = unit(eig(X)); % unit l 176. eig : calcul valeur propre de X
     i        = find( abs(dg) < epsil ); % i contient les indices ik de dg tel que pour tout k, dg(ik)<epsil
     dg(i)    = 0;
     nneg     = length(find( dg < -epsil)); % cardinal du sous ensemble d'indices vÃ©rifiant la condition dg<-epsil
     npos     = length(find( dg >  epsil ));
     s1       = max(nneg,npos); % maximum
     s2       = min(nneg,npos);
     if nargout<=1, s1 = [s1 s2]; end;
return;
*/
template<class MatA>
void signature( const MatA & x, std::size_t & s1, std::size_t & s2, const double epsilon = 10e-8 )
{
	namespace ublas = boost::numeric::ublas;

	typedef typename MatA::value_type T;
	ublas::matrix<T> xn = x / rom::numerical::norm2( x );
	ublas::diagonal_matrix<T> d;
	rom::numerical::eigd( xn, d );
	ublas::vector<T> dg = unit( d );
	typename ublas::vector<T>::iterator it;
	std::size_t nneg = 0;
	std::size_t npos = 0;
	for( it = dg.begin(); it != dg.end(); ++it )
	{
		if ( std::abs(*it) < epsilon )
		{
			*it = 0;
		}
		else if ( *it < -epsilon )
		{
			++nneg;
		}
		else if ( *it > epsilon )
		{
			++npos;
		}
	}
	s1 = std::max( nneg, npos );
	s2 = std::min( nneg, npos );
}

void get2CoplanarCircleConstraint( const boost::numeric::ublas::bounded_matrix<double, 3, 3> & a,
	                               const boost::numeric::ublas::bounded_matrix<double, 3, 3> & b,
                                   boost::numeric::ublas::matrix<double> & M, std::size_t k );

void rectifyHomography2PlanarCC(const rom::numerical::geometry::Ellipse & e11, const rom::numerical::geometry::Ellipse & e21, boost::numeric::ublas::bounded_matrix<double, 3, 3> & mH, const double distMarkers = 1.0);

void homographyFrom2CPlanar( const std::pair<CCTag, CCTag> & cctags, boost::numeric::ublas::bounded_matrix<double, 3, 3> & h);

bool disambiguate(const rom::numerical::geometry::Ellipse & q, const boost::numeric::ublas::bounded_matrix<double, 3, 3> & H);


}
}
}


#endif
