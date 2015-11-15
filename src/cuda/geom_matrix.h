#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

namespace popart {
namespace geometry {

class matrix3x3
{
    float val[3][3];

public:
    __host__ __device__
    matrix3x3( ) { }

    // Note: default copy contructor and
    //       default assignment operator
    //       are allowed (and required)

    __host__ __device__
    matrix3x3( const float mx[3][3] );

    __host__ __device__
    matrix3x3( float m00, float m01, float m02,
               float m10, float m11, float m12,
               float m20, float m21, float m22 )
    {
        val[0][0] = m00;
        val[0][1] = m01;
        val[0][2] = m02;
        val[1][0] = m10;
        val[1][1] = m11;
        val[1][2] = m12;
        val[2][0] = m20;
        val[2][1] = m21;
        val[2][2] = m22;
    }

    __host__ __device__
    void setDiag( float v00, float v11, float v22 );

    __host__ __device__
    void clear( );

    __host__ __device__
    inline float& operator()(int y, int x) {
        return val[y][x];
    }

    __host__ __device__
    inline const float& operator()(int y, int x) const {
        return val[y][x];
    }

    __host__ __device__
    float det( ) const;

    __host__ __device__
    bool invert( matrix3x3& result ) const;

    __host__ __device__
    float2 applyHomography( const float2& vec ) const;

    __host__ __device__
    float2 applyHomography( float x, float y ) const;

    __host__ __device__
    void condition( float2& homVec ) const;

    __host__
    inline void print( std::ostream& ostr ) const {
        ostr << "r1=(" << val[0][0] << "," << val[0][1] << "," << val[0][2] << ")"
             << " r2=(" << val[1][0] << "," << val[1][1] << "," << val[1][2] << ")"
             << " r3=(" << val[2][0] << "," << val[2][1] << "," << val[2][2] << ")";
    }
    __device__
    inline void printf( char* buffer ) const {
        ::sprintf( buffer, "r1=(%f,%f,%f) r2=(%f,%f,%f) r3=(%f,%f,%f)",
                   val[0][0], val[0][1], val[0][2],
                   val[1][0], val[1][1], val[1][2],
                   val[2][0], val[2][1], val[2][2] );
    }
};

class matrix3x3_tView
{
    const matrix3x3& _m;
public:

    __host__ __device__
    matrix3x3_tView( const matrix3x3& m )
        : _m(m)
    { }

    __host__ __device__
    inline const float& operator()(int y, int x) const {
        return _m.operator()(x,y);
    }

    __host__ __device__
    inline float det( ) const {
        return _m.det();
    }

private:
    // forbidden default constructor and assignment op

    __host__ __device__
    matrix3x3_tView( );

    __host__ __device__
    matrix3x3_tView( const matrix3x3_tView& );

    __host__ __device__
    matrix3x3_tView& operator=( const matrix3x3_tView& );
};

__host__ __device__
float2 prod_normvec2normvec( const matrix3x3& l, const float2& r );

__host__ __device__
float3 prod_normvec2vec( const matrix3x3& l, const float2& r );

__host__ __device__
matrix3x3 prod( const matrix3x3& l,       const matrix3x3& r );

__host__ __device__
matrix3x3 prod( const matrix3x3_tView& l, const matrix3x3& r );

__host__ __device__
matrix3x3 prod( const matrix3x3& l,       const matrix3x3_tView& r );

}; // namespace geometry
}; // namespace popart

