/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "geom_matrix.h"

using namespace std;

namespace cctag {
namespace geometry {

__host__ __device__
matrix3x3::matrix3x3( const float mx[3][3] )
{
    #pragma unroll
    for( int i=0; i<3; i++ ) {
        #pragma unroll
        for( int j=0; j<3; j++ ) {
            val[i][j] = mx[i][j];
        }
    }
}

__host__ __device__
void matrix3x3::setDiag( float v00, float v11, float v22 )
{
    clear();
    val[0][0] = v00;
    val[1][1] = v11;
    val[2][2] = v22;
}

__host__ __device__
void matrix3x3::clear( )
{
    #pragma unroll
    for( int i=0; i<3; i++ ) {
        #pragma unroll
        for( int j=0; j<3; j++ ) {
            val[i][j] = 0;
        }
    }
}


__device__
float2 matrix3x3::applyHomography( const float2& vec ) const
{
    float u = val[0][0]*vec.x + val[0][1]*vec.y + val[0][2];
    float v = val[1][0]*vec.x + val[1][1]*vec.y + val[1][2];
    float w = val[2][0]*vec.x + val[2][1]*vec.y + val[2][2];
    float2 result; //  = make_float2( u/w, v/w );
    result.x = u/w;
    result.y = v/w;
    return result;
}

__device__
float2 matrix3x3::applyHomography( float x, float y ) const
{
    float u = val[0][0]*x + val[0][1]*y + val[0][2];
    float v = val[1][0]*x + val[1][1]*y + val[1][2];
    float w = val[2][0]*x + val[2][1]*y + val[2][2];
    float2 result; //  = make_float2( u/w, v/w );
    result.x = u/w;
    result.y = v/w;
    return result;
}

__host__ __device__
void matrix3x3::condition( float2& homVec ) const
{
    homVec = prod_normvec2normvec( *this, homVec );
}

__host__ __device__
matrix3x3 prod( const matrix3x3& l, const matrix3x3& r )
{
    matrix3x3 result;
    #pragma unroll
    for( int y=0; y<3; y++ ) {
        #pragma unroll
        for( int x=0; x<3; x++ ) {
            result(y,x) = l(y,0)*r(0,x) + l(y,1)*r(1,x) + l(y,2)*r(2,x);
        }
    }
    return result;
}

__host__ __device__
float2 prod_normvec2normvec( const matrix3x3& l, const float2& r )
{
    float2 d12;
    float  d3;

    d12.x  = l(0,0)*r.x + l(0,1)*r.y + l(0,2);
    d12.y  = l(1,0)*r.x + l(1,1)*r.y + l(1,2);
    d3     = l(2,0)*r.x + l(2,1)*r.y + l(2,2);
#ifdef __CUDA_ARCH__
    if( d3 != 0.0f ) {
        d3     = __frcp_rn( d3 );
        d12.x *= d3;
        d12.y *= d3;
        return d12;
    } else {
        return make_float2( 0.0f, 0.0f );
    }
#else // not __CUDA_ARCH__
    if( d3 != 0.0f ) {
        d12.x /= d3;
        d12.y /= d3;
        return d12;
    } else {
        cerr << __FILE__ << ":" << __LINE__
             << "matrix X normalized vector -> scale is 0" << endl;
        return make_float2( 0.0f, 0.0f );
    }
#endif // not __CUDA_ARCH__
}

__host__ __device__
float3 prod_normvec2vec( const matrix3x3& l, const float2& r )
{
    float3 d;
    d.x = l(0,0)*r.x + l(0,1)*r.y + l(0,2);
    d.y = l(1,0)*r.x + l(1,1)*r.y + l(1,2);
    d.z = l(2,0)*r.x + l(2,1)*r.y + l(2,2);
    return d;
}

__host__ __device__
matrix3x3 prod( const matrix3x3_tView& l, const matrix3x3& r )
{
    matrix3x3 result;
    #pragma unroll
    for( int y=0; y<3; y++ ) {
        #pragma unroll
        for( int x=0; x<3; x++ ) {
            result(y,x) = l(y,0)*r(0,x) + l(y,1)*r(1,x) + l(y,2)*r(2,x);
        }
    }
    return result;
}

__host__ __device__
matrix3x3 prod( const matrix3x3& l, const matrix3x3_tView& r )
{
    matrix3x3 result;
    #pragma unroll
    for( int y=0; y<3; y++ ) {
        #pragma unroll
        for( int x=0; x<3; x++ ) {
            result(y,x) = l(y,0)*r(0,x) + l(y,1)*r(1,x) + l(y,2)*r(2,x);
        }
    }
    return result;
}

}; // namespace geometry
}; // namespace cctag

