#pragma once

#include <cuda_runtime.h>
#include "geom_matrix.h"

namespace popart {
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

    __host__ __device__
	ellipse transform(const matrix3x3& mT)  const;

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

	/// @todo: is it the correct name ??
	// inline bounded_vector<double, 6> colon() const;

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
}; // namespace popart

