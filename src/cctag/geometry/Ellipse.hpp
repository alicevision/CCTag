#ifndef _CCTAG_NUMERICAL_ELLIPSE_HPP_
#define _CCTAG_NUMERICAL_ELLIPSE_HPP_

#include <cctag/geometry/point.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/functional.hpp>

#include <iostream>

namespace cctag {
namespace numerical {
namespace geometry {

using boost::numeric::ublas::bounded_matrix;
using boost::numeric::ublas::bounded_vector;

class Ellipse
{
public:
	typedef boost::numeric::ublas::bounded_matrix<double, 3, 3> Matrix;
	Ellipse()
		: _a( 0.0 )
		, _b( 0.0 )
		, _angle( 0.0 )
	{
		_center.clear();
		_matrix.clear();
	}

	Ellipse( const Matrix& matrix );
	Ellipse( const Point2dN<double>& center, const double a, const double b, const double angle );

	virtual ~Ellipse() {}

	inline const Matrix& matrix() const { return _matrix; }
	inline Matrix& matrix() { return _matrix; }
	inline const Point2dN<double>& center() const { return _center; }
	inline Point2dN<double>& center() { return _center; }
	inline double a() const      { return _a; }
	inline double b() const      { return _b; }
	inline double angle() const  { return _angle; }

	void setMatrix( const Matrix& matrix );

	void setParameters( const Point2dN<double>& center, const double a, const double b, const double angle );

	void setCenter( const Point2dN<double>& center );

	void setA( const double a );

	void setB( const double b );

	void setAngle( const double angle );

	Ellipse transform(const Matrix& mT) const;

	void computeParameters();

	void computeMatrix();
        
        void getCanonicForm(Matrix& mCanonic, Matrix& mTprimal, Matrix& mTdual) const;

	void init( const Point2dN<double>& center, const double a, const double b, const double angle );

	/// @todo: is it the correct name ??
	inline bounded_vector<double, 6> colon() const;

	friend  std::ostream& operator<<(std::ostream& os, const Ellipse& e);

protected:
	Matrix _matrix;
	Point2dN<double> _center;
	double _a;
	double _b;
	double _angle;
};

void getSortedOuterPoints(
        const Ellipse & ellipse,
        const std::vector< cctag::DirectedPoint2d<double> > & points,
        std::vector< cctag::DirectedPoint2d<double> > & resPoints,
        const std::size_t requestedSize);

inline bounded_vector<double, 6> Ellipse::colon() const
{
	bounded_vector<double, 6> qColon;
	qColon( 0 ) = _matrix( 0, 0 );
	qColon( 1 ) = _matrix( 0, 1 );
	qColon( 2 ) = _matrix( 0, 2 );
	qColon( 3 ) = _matrix( 1, 1 );
	qColon( 4 ) = _matrix( 1, 2 );
	qColon( 5 ) = _matrix( 2, 2 );
	return qColon;
}


void scale(const Ellipse & ellipse, Ellipse & rescaleEllipse, double scale);

}
}
}

#endif

