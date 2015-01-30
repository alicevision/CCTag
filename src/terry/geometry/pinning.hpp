#ifndef _TERRY_GEOMETRY_PINNING_HPP_
#define	_TERRY_GEOMETRY_PINNING_HPP_

#include <boost/numeric/ublas/matrix.hpp>

namespace terry {
namespace geometry {

template<typename Scalar>
struct PinningPerspective
{
	double _width, _height;
	boost::numeric::ublas::bounded_matrix<Scalar,3,3> _matrix;
};

template<typename Scalar>
struct PinningBilinear
{
	double _width, _height;
	boost::numeric::ublas::bounded_matrix<Scalar,2,4> _matrix;
};


/**
 * @brief Perspective transformation functor.
 * @param[in] t: the transformation 3x3 matrix
 * @param[in] src: 2D source point
 */
template <typename F, typename F2>
inline boost::gil::point2<F> transform( const PinningPerspective<F>& t, const boost::gil::point2<F2>& src )
{
	using namespace boost::numeric::ublas;
        bounded_vector<F,3> pIn;

	F hCenter = ((0.5*t._height)/t._width); ///@todo tuttle: modify the matrix instead

	pIn[0] = (src.x / t._width) - 0.5;
	pIn[1] = (src.y / t._width) - hCenter;
	pIn[2] = 1.0;

	bounded_vector<F,3> pOut = prod( t._matrix, pIn );

	boost::gil::point2<F> res;
	res.x = pOut[0] / pOut[2];
	res.y = pOut[1] / pOut[2];
	//res.x = (t._matrix(0, 0) * pIn[0] + t._matrix(0, 1) * pIn[1] + t._matrix(0, 2)) / (t._matrix(2, 0) * pIn[0] + t._matrix(2, 1) * pIn[1] + t._matrix(2, 2));
	//res.y = (t._matrix(1, 0) * pIn[0] + t._matrix(1, 1) * pIn[1] + t._matrix(1, 2)) / (t._matrix(2, 0) * pIn[0] + t._matrix(2, 1) * pIn[1] + t._matrix(2, 2));

	res.x = (res.x + 0.5) * t._width;
	res.y = (res.y + hCenter) * t._width;
	
	return res;
}

/**
 * @brief Bilinear transformation functor.
 * @param[in] t: the transformation 2x4 matrix
 * @param[in] src: 2D source point
 *
 * @f[
 * x' = c[0,0]x + c[0,1]y + c[0,2]xy + c[0,3]
 * y' = c[1,0]x + c[1,1]y + c[1,2]xy + c[1,3]
 * @f]
 */
template <typename F, typename F2>
inline boost::gil::point2<F> transform( const PinningBilinear<F>& t, const boost::gil::point2<F2>& src )
{
	boost::gil::point2<F> res;

	F hCenter = ((0.5*t._height)/t._width);
	boost::gil::point2<F> in( (src.x / t._width) - 0.5, (src.y / t._width) - hCenter );

	res.x = t._matrix(0, 0) * in.x + t._matrix(0, 1) * in.y + t._matrix(0, 2) * in.x * in.y + t._matrix(0, 3);
	res.y = t._matrix(1, 0) * in.x + t._matrix(1, 1) * in.y + t._matrix(1, 2) * in.x * in.y + t._matrix(1, 3);

	res.x = (res.x + 0.5) * t._width;
	res.y = (res.y + hCenter) * t._width;
	return res;
}

}
}

#endif

