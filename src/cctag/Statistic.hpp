#ifndef _CCTAG_STATISTIC_HPP_
#define _CCTAG_STATISTIC_HPP_

#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/units/cmath.hpp>
#include <boost/swap.hpp>

#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>
#include <set>
#include <algorithm>
#include <cassert>

namespace cctag {
namespace numerical {

namespace ublas = boost::numeric::ublas;

/**
 * Compute a random permutation of the integers from 1 to n
 * @param n
 * @return
 */
template<class V>
V randperm( const std::size_t n )
{
	V temp( n );
	for( std::size_t i = 0; i < n; ++i )
	{
		temp[ i ] = i;
	}

	for( std::size_t i = 0; i < 2 * n; ++i )
	{
		std::size_t i1  = std::rand() % n;
		const std::size_t i2  = std::rand() % n;
		if ( i1 == i2 )
		{
			continue;
		}
		boost::swap( temp[i1], temp[i2] );
	}
	return temp;
}

// Draw N unique values in the range of 0 .. (K-1)
// and copy them in the container
template <typename Container>
void rand_n_k(Container &container, size_t N, size_t K)
{
    // If the number of element is small, we 
    // might just try to do a linear search for values
    // already drawn instead of using a set
    std::set<size_t> values; 
    assert(K>N);
    while(values.size()<N)
    {
        values.insert(rand()%K);
    }  
       
    container.resize(N);
    // Note that the values are orderer, if you want them in random order
    // use random_shuffle
    copy(values.begin(), values.end(), container.begin());
}


// median(X) is the median value of the elements in X.
double median( std::vector<double>& v );

// Compute the mean of a vector of bounded_vector<double,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<double, 3> mean( const V& v )
{
	ublas::bounded_vector<double, 3> mv;
	mv( 0 ) = 0.0;
	mv( 1 ) = 0.0;

	for( typename V::const_iterator it = v.begin(); it != v.end() ; ++it )
	{
		mv( 0 ) += (*it)( 0 );
		mv( 1 ) += (*it)( 1 );
	}

	mv( 0 ) /= v.size();
	mv( 1 ) /= v.size();
	mv( 2 ) = 1.0;

	return mv;
}

// Compute the standard deviation of a vector of bounded_vector<double,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<double, 3> stdDev( const V& v )
{
	ublas::bounded_vector<double, 3> mv  = mean( v );
	ublas::bounded_vector<double, 3> var = stdDev( v, mv );
	return var;
}

// Compute the standard deviation of a vector of bounded_vector<double,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<double, 3> stdDev( const V& v, const ublas::bounded_vector<double, 3>& mv )
{
	ublas::bounded_vector<double, 3> var;
	var( 0 ) = 0;
	var( 1 ) = 0;

	for( typename V::const_iterator it = v.begin(); it != v.end() ; ++it )
	{
		var( 0 ) += ( mv( 0 ) - (*it)( 0 ) ) * ( mv( 0 ) - (*it)( 0 ) );
		var( 1 ) += ( mv( 1 ) - (*it)( 1 ) ) * ( mv( 1 ) - (*it)( 1 ) );
	}

	var( 0 ) = sqrt( var( 0 ) / v.size() );
	var( 1 ) = sqrt( var( 1 ) / v.size() );
	var( 2 ) = 1.0;

	return var;
}

template<class V>
inline double median( V v )
{
	std::sort( v.begin(), v.end() );
	return v[v.size() / 2];
}

template<class V>
inline double medianRef( V & v )
{
	std::sort( v.begin(), v.end() );
	return v[v.size() / 2];
}


} // namespace numerical
} // namespace cctag

#endif

