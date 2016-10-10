/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_STATISTIC_HPP_
#define _CCTAG_STATISTIC_HPP_

#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>
#include <set>
#include <algorithm>
#include <cassert>
#include <array>

namespace cctag {
namespace numerical {

/**
 * Compute a random permutation of the integers from 1 to n
 * @param n
 * @return
 */
#if 0 // Currently unused, otherwise use std::random_shuffle
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
#endif

// Draw N unique values in the range of 0 .. (K-1)
// and copy them in the container
#if 0
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
#endif

void rand_5_k(std::array<int, 5>& perm, size_t N);

// median(X) is the median value of the elements in X.
float median( std::vector<float>& v );

#if 0
// Compute the mean of a vector of bounded_vector<float,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<float, 3> mean( const V& v )
{
	ublas::bounded_vector<float, 3> mv;
	mv( 0 ) = 0.f;
	mv( 1 ) = 0.f;

	for( typename V::const_iterator it = v.begin(); it != v.end() ; ++it )
	{
		mv( 0 ) += (*it)( 0 );
		mv( 1 ) += (*it)( 1 );
	}

	mv( 0 ) /= v.size();
	mv( 1 ) /= v.size();
	mv( 2 ) = 1.f;

	return mv;
}

// Compute the standard deviation of a vector of bounded_vector<float,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<float, 3> stdDev( const V& v )
{
	ublas::bounded_vector<float, 3> mv  = mean( v );
	ublas::bounded_vector<float, 3> var = stdDev( v, mv );
	return var;
}

// Compute the standard deviation of a vector of bounded_vector<float,3>* considered as Point2dN (i.e. of size 2)
template<class V>
ublas::bounded_vector<float, 3> stdDev( const V& v, const ublas::bounded_vector<float, 3>& mv )
{
	ublas::bounded_vector<float, 3> var;
	var( 0 ) = 0;
	var( 1 ) = 0;

	for( typename V::const_iterator it = v.begin(); it != v.end() ; ++it )
	{
		var( 0 ) += ( mv( 0 ) - (*it)( 0 ) ) * ( mv( 0 ) - (*it)( 0 ) );
		var( 1 ) += ( mv( 1 ) - (*it)( 1 ) ) * ( mv( 1 ) - (*it)( 1 ) );
	}

	var( 0 ) = sqrt( var( 0 ) / v.size() );
	var( 1 ) = sqrt( var( 1 ) / v.size() );
	var( 2 ) = 1.f;

	return var;
}
#endif

template<class V>
inline float median( V v )
{
	std::sort( v.begin(), v.end() );
	return v[v.size() / 2];
}

template<class V>
inline float medianRef( V & v )
{
	std::sort( v.begin(), v.end() );
	return v[v.size() / 2];
}


} // namespace numerical
} // namespace cctag

#endif

