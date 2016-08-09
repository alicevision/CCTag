#ifndef _CCTAG_NUMERICAL_ALGEBRA_OPERATION_HPP_
#define	_CCTAG_NUMERICAL_ALGEBRA_OPERATION_HPP_

#include <Eigen/Core>
#include <cmath>
#include "../../utils/Exceptions.hpp"

namespace cctag {
namespace numerical {

Eigen::Matrix3f& normalizeDet1( Eigen::Matrix3f& m );

#if 0
template<class Matrix>
typename Matrix::value_type trace(const Matrix& m)
{
    typedef typename Matrix::value_type T;
    T tr = 0;
    std::size_t n = std::min( m.size1(), m.size2());
    for(std::size_t i = 0 ; i < n ; ++i )
    {
        tr += m(i,i);
    }
    return tr;
}


template<class M>
M& matSqrt( M& S )
{
	for( std::size_t k = 0; k < S.size1(); ++k )
	{
		S(k, k) = std::sqrt( S(k, k) );
	}
	return S;
}

/*
template<typename T>
inline std::complex<T> product( const std::complex<T> & a, const std::complex<T> & b )
{
	return std::complex<T>();
}
*/

template<class V>
inline cctag::numerical::BoundedVector3d cross(const V & vX, const V & vY)
{
	BoundedVector3d res;
	res(0) = vX(1) * vY(2) - vX(2) * vY(1);
	res(1) = vX(2) * vY(0) - vX(0) * vY(2);
	res(2) = vX(0) * vY(1) - vX(1) * vY(0);
	return res;
}

template<class T>
inline cctag::numerical::BoundedVector3d unit(const T & v)
{
	using namespace boost::numeric::ublas;
	return v/norm_2(v);
}

template<class V>
inline V normalize(const V & v)
{
	BOOST_ASSERT( v.size() == 3 );
	if ( v( 2 ) == 0 )
	{
		BOOST_THROW_EXCEPTION(
			exception::Bug()
			<< exception::dev() + "Normalization of an infinite point !" );
	}
	V ret = v;
	ret( 0 ) /= ret( 2 );
	ret( 1 ) /= ret( 2 );
	ret( 2 ) = 1.f;
	return ret;
}
#endif
}
}


#endif

