/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include "geom_matrix.h"

namespace cctag {
namespace geometry {

struct ellipse
{
public:
    __host__ __device__
	ellipse()
		: _a( 0.0f )
		, _b( 0.0f )
		, _angle( 0.0f )
	{
        _center = make_float2(0.0f,0.0f);
		_matrix.clear();
	}

    __host__ __device__
    ellipse( float m00, float m01, float m02,
             float m10, float m11, float m12,
             float m20, float m21, float m22,
             float centerx, float centery,
             float a, float b, float angle )
        : _matrix( m00, m01, m02,
                   m10, m11, m12,
                   m20, m21, m22 )
        , _a(a)
        , _b(b)
        , _angle(angle)
    {
        _center = make_float2( centerx, centery );
    }

    __host__ __device__
	ellipse( const matrix3x3& matrix );

    __host__ __device__
	ellipse( const float2& center, const float a, const float b, const float angle );

    // __host__ __device__ virtual ~ellipse() {}

    __host__ __device__
	inline const matrix3x3& matrix() const { return _matrix; }

    __host__ __device__
	inline matrix3x3& matrix() { return _matrix; }

    __host__ __device__
	inline const float2& center() const { return _center; }

    __host__ __device__
	inline float2& center() { return _center; }

    __host__ __device__
	inline float a() const      { return _a; }

    __host__ __device__
	inline float b() const      { return _b; }

    __host__ __device__
	inline float angle() const  { return _angle; }

    __host__ __device__
	void setMatrix( const matrix3x3& matrix );

    __host__ __device__
	void setParameters( const float2& center, const float a, const float b, const float angle );

    __host__ __device__
	void setCenter( const float2& center );

    __host__ __device__
	void setA( const float a );

    __host__ __device__
	void setB( const float b );

    __host__ __device__
	void setAngle( const float angle );

#if 0
    __host__ __device__
	ellipse transform(const matrix3x3& mT)  const;
#endif

    __host__ __device__
    void projectiveTransform( const matrix3x3& m, ellipse& e ) const;

    // two separate implementations!
    __host__ __device__
	void computeParameters();

    // two separate implementations!
    __host__ __device__
	void computeMatrix();
        
    __device__
    void getCanonicForm(matrix3x3& mCanonic, matrix3x3& mTprimal, matrix3x3& mTdual) const;

    __host__ __device__
	inline void init( const float2& center, const float a, const float b, const float angle ) {
        setParameters( center, a, b, angle );
    }

    __device__
    void computeHomographyFromImagedCenter( const float2 center, matrix3x3& mHomography ) const;

    __host__ __device__
    void makeConditionerFromEllipse( matrix3x3& output ) const;

private:
	matrix3x3 _matrix;
	float2    _center;
	float     _a;
	float     _b;
	float     _angle;

    __host__ __device__
    void crash( const char* file, int line, const char* msg );
};

}; // namespace geometry
}; // namespace cctag

