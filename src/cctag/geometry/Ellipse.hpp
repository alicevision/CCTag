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
	typedef Eigen::Matrix3f Matrix;
        
	Ellipse()
                : _matrix(Eigen::Matrix3f::Zero())
                , _center(0, 0)
		, _a( 0.0 )
		, _b( 0.0 )
		, _angle( 0.0 )
	{
	}

	Ellipse( const Matrix& matrix );
	Ellipse( const Point2d<Eigen::Vector3f>& center, const double a, const double b, const double angle );

	inline const Matrix& matrix() const { return _matrix; }
	inline Matrix& matrix() { return _matrix; }
	inline const Point2d<Eigen::Vector3f>& center() const { return _center; }
	inline Point2d<Eigen::Vector3f>& center() { return _center; }
	inline double a() const      { return _a; }
	inline double b() const      { return _b; }
	inline double angle() const  { return _angle; }

	void setMatrix( const Matrix& matrix );

	void setParameters( const Point2d<Eigen::Vector3f>& center, const double a, const double b, const double angle );

	void setCenter( const Point2d<Eigen::Vector3f>& center );

	void setA( const double a );

	void setB( const double b );

	void setAngle( const double angle );

	Ellipse transform(const Matrix& mT) const;

	void computeParameters();

	void computeMatrix();
        
        void getCanonicForm(Matrix& mCanonic, Matrix& mTprimal, Matrix& mTdual) const;

	void init( const Point2d<Eigen::Vector3f>& center, const double a, const double b, const double angle );

	friend  std::ostream& operator<<(std::ostream& os, const Ellipse& e);

protected:
	Eigen::Matrix3f _matrix;
	Point2d<Eigen::Vector3f> _center;
	double _a;
	double _b;
	double _angle;
};

void getSortedOuterPoints(
        const Ellipse & ellipse,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & points,
        std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & resPoints,
        const std::size_t requestedSize);

void scale(const Ellipse & ellipse, Ellipse & rescaleEllipse, double scale);

}
}
}

#endif

