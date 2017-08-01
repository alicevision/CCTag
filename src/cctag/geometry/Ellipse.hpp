/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_NUMERICAL_ELLIPSE_HPP_
#define _CCTAG_NUMERICAL_ELLIPSE_HPP_

#include <cctag/geometry/Point.hpp>
#include <Eigen/Core>
#include <iostream>

namespace cctag {
namespace numerical {
namespace geometry {

class Ellipse
{
public:
	using Matrix = Eigen::Matrix3f;
        
	Ellipse()
                : _matrix(Eigen::Matrix3f::Zero())
                , _center(0, 0)
		, _a( 0.f )
		, _b( 0.f )
		, _angle( 0.f )
	{
	}

	Ellipse( const Matrix& matrix );
	Ellipse( const Point2d<Eigen::Vector3f>& center, const float a, const float b, const float angle );

	inline const Matrix& matrix() const { return _matrix; }
	inline Matrix& matrix() { return _matrix; }
	inline const Point2d<Eigen::Vector3f>& center() const { return _center; }
	inline Point2d<Eigen::Vector3f>& center() { return _center; }
	inline float a() const      { return _a; }
	inline float b() const      { return _b; }
	inline float angle() const  { return _angle; }

	void setMatrix( const Matrix& matrix );

	void setParameters( const Point2d<Eigen::Vector3f>& center, const float a, const float b, const float angle );

	void setCenter( const Point2d<Eigen::Vector3f>& center );

	void setA( const float a );

	void setB( const float b );

	void setAngle( const float angle );

	Ellipse transform(const Matrix& mT) const;

	void computeParameters();

	void computeMatrix();
        
        void getCanonicForm(Matrix& mCanonic, Matrix& mTprimal, Matrix& mTdual) const;

	void init( const Point2d<Eigen::Vector3f>& center, const float a, const float b, const float angle );

	friend  std::ostream& operator<<(std::ostream& os, const Ellipse& e);

protected:
	Eigen::Matrix3f _matrix;
	Point2d<Eigen::Vector3f> _center;
	float _a;
	float _b;
	float _angle;
};

void getSortedOuterPoints(
        const Ellipse & ellipse,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & points,
        std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & resPoints,
        const std::size_t requestedSize);

void scale(const Ellipse & ellipse, Ellipse & rescaleEllipse, float scale);

}
}
}

#endif

