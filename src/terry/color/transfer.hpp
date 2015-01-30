#ifndef _TERRY_COLOR_COLORTRANSFER_HPP_
#define _TERRY_COLOR_COLORTRANSFER_HPP_

#include <terry/numeric/matrix.hpp>
#include <terry/numeric/pow.hpp>
#include <terry/numeric/log.hpp>
#include <terry/numeric/clamp.hpp>
#include <terry/typedefs.hpp>
#include <terry/channel.hpp>

//#include <boost/gil/extension/color/hsl.hpp>
//#include <boost/gil/extension/color/distribution.hpp>

namespace terry {
namespace color {
namespace transfer {

namespace detail {
template<typename T>
struct matrices_rgb_to_lab_t
{
	typedef boost::numeric::ublas::bounded_matrix<T, 3, 3 > Matrix33;
	Matrix33 _RGB_2_LMS;
	Matrix33 _LMS_2_LAB;

	matrices_rgb_to_lab_t()
	: _RGB_2_LMS(3,3)
	, _LMS_2_LAB(3,3)
	{
		_RGB_2_LMS(0, 0) =  0.3811;
		_RGB_2_LMS(0, 1) =  0.5783;
		_RGB_2_LMS(0, 2) =  0.0402;
		_RGB_2_LMS(1, 0) =  0.1967;
		_RGB_2_LMS(1, 1) =  0.7244;
		_RGB_2_LMS(1, 2) =  0.0782;
		_RGB_2_LMS(2, 0) =  0.0241;
		_RGB_2_LMS(2, 1) =  0.1288;
		_RGB_2_LMS(2, 2) =  0.8444;

		// LMS -> LAB
		// [L]   [ 1/sqrt(3)     0         0     ] [ 1  1  1 ] [L]
		// [A] = [    0      1/sqrt(6)     0     ] [ 1  1 -2 ] [M]
		// [B]   [    0          0     1/sqrt(2) ] [ 1 -1  0 ] [S]
		//
		// [L]   [ 1/sqrt(3)  1/sqrt(3)  1/sqrt(3) ] [L]
		// [A] = [ 1/sqrt(6)  1/sqrt(6) -2/sqrt(6) ] [M]
		// [B]   [ 1/sqrt(2) -1/sqrt(2)     0.0    ] [S]
		static const T invSqrt2 = 1.0 / std::sqrt( (double) 2.0 );
		static const T invSqrt3 = 1.0 / std::sqrt( (double) 3.0 );
		static const T invSqrt6 = 1.0 / std::sqrt( (double) 6.0 );
		_LMS_2_LAB(0, 0) =  invSqrt3;
		_LMS_2_LAB(0, 1) =  invSqrt3;
		_LMS_2_LAB(0, 2) =  invSqrt3;
		_LMS_2_LAB(1, 0) =  invSqrt6;
		_LMS_2_LAB(1, 1) =  invSqrt6;
		_LMS_2_LAB(1, 2) = -2.0 * invSqrt6;
		_LMS_2_LAB(2, 0) =  invSqrt2;
		_LMS_2_LAB(2, 1) = -invSqrt2;
		_LMS_2_LAB(2, 2) =  0.0;
	}
};
template<typename T>
struct matrices_lab_to_rgb_t
{
	typedef boost::numeric::ublas::bounded_matrix<T, 3, 3 > Matrix33;
	Matrix33 _LAB_2_LMS;
	Matrix33 _LMS_2_RGB;

	matrices_lab_to_rgb_t()
	: _LAB_2_LMS(3,3)
	, _LMS_2_RGB(3,3)
	{
		// LAB -> LMS
		// [L]   [ 1  1  1 ] [ sqrt(3)/3     0         0     ] [L]
		// [M] = [ 1  1 -1 ] [    0      sqrt(6)/6     0     ] [A]
		// [S]   [ 1 -2  0 ] [    0          0     sqrt(2)/2 ] [B]
		//
		// [L]   [ sqrt(3)/3   sqrt(6)/6   sqrt(2)/2 ] [L]
		// [M] = [ sqrt(3)/3   sqrt(6)/6  -sqrt(2)/2 ] [A]
		// [S]   [ sqrt(3)/3 -2*sqrt(6)/6       0    ] [B]
		static const T sqrt2_2 = 1.0 / std::sqrt( (double) 2.0 );
		static const T sqrt3_3 = 1.0 / std::sqrt( (double) 3.0 );
		static const T sqrt6_6 = 1.0 / std::sqrt( (double) 6.0 );
		_LAB_2_LMS(0, 0) =  sqrt3_3;
		_LAB_2_LMS(0, 1) =  sqrt6_6;
		_LAB_2_LMS(0, 2) =  sqrt2_2;
		_LAB_2_LMS(1, 0) =  sqrt3_3;
		_LAB_2_LMS(1, 1) =  sqrt6_6;
		_LAB_2_LMS(1, 2) = -sqrt2_2;
		_LAB_2_LMS(2, 0) =  sqrt3_3;
		_LAB_2_LMS(2, 1) = -2.0 * sqrt3_3;
		_LAB_2_LMS(2, 2) =  0.0;

		_LMS_2_RGB(0, 0) =  4.4679;
		_LMS_2_RGB(0, 1) = -3.5873;
		_LMS_2_RGB(0, 2) =  0.1193;
		_LMS_2_RGB(1, 0) = -1.2186;
		_LMS_2_RGB(1, 1) =  2.3809;
		_LMS_2_RGB(1, 2) = -0.1624;
		_LMS_2_RGB(2, 0) =  0.0497;
		_LMS_2_RGB(2, 1) = -0.2439;
		_LMS_2_RGB(2, 2) =  1.2045;

	}
};
}

template <typename PixelRef, typename PixelR = PixelRef> // models pixel concept
struct pixel_rgb_to_lab_t
{
	typedef typename channel_type<PixelR>::type ChannelR;
	typedef typename floating_channel_type_t<ChannelR>::type T;
	typedef typename detail::matrices_rgb_to_lab_t<T> MatrixContants;
	typedef typename MatrixContants::Matrix33 Matrix;

	static const MatrixContants _matrices;

	GIL_FORCEINLINE
	PixelR operator()( const PixelRef & rgb ) const
	{
		using namespace terry::numeric;
		static const T thresold = 1.0e-5;
		static const ChannelR channelThresold = ChannelR(thresold); //channel_convert<ChannelR, T>( thresold );
		// RGB to LMS
		PixelR lms = pixel_matrix33_multiply_t<PixelRef, Matrix, PixelR>( _matrices._RGB_2_LMS )( rgb );
		// log(v)
		PixelR lms_log = pixel_log10_t<PixelR, PixelR>()(
				// log10(x) is not defined for x <= 0, so we need to clamp
				pixel_clamp_lower_than_t<PixelR>( channelThresold, channelThresold )(
					lms
				)
			);
		// LMS to LAB (lambda alpha beta)
		PixelR lab = pixel_matrix33_multiply_t<PixelR, Matrix, PixelR>( _matrices._LMS_2_LAB )( lms_log );
		return lab;
	}
};

template <typename PixelRef, typename PixelR>
const typename pixel_rgb_to_lab_t<PixelRef, PixelR>::MatrixContants pixel_rgb_to_lab_t<PixelRef, PixelR>::_matrices; // init static variable.


template <typename PixelRef, typename PixelR = PixelRef> // models pixel concept
struct pixel_lab_to_rgb_t
{
	typedef typename channel_type<PixelR>::type ChannelR;
	typedef typename floating_channel_type_t<ChannelR>::type T;
	typedef typename detail::matrices_lab_to_rgb_t<T> MatrixContants;
	typedef typename MatrixContants::Matrix33 Matrix;

	static const MatrixContants _matrices;

	GIL_FORCEINLINE
	PixelR operator()( const PixelRef & lab ) const
	{
		using namespace terry::numeric;

		// LAB (lambda alpha beta) to LMS
		PixelR lms_log = pixel_matrix33_multiply_t<PixelRef, Matrix, PixelR>( _matrices._LAB_2_LMS )( lab );
		// 10^v
		PixelR lms = pixel_scalar_pow_t<PixelR, T, PixelR>()( lms_log, 10.0 );
		// LMS to RGB
		PixelR rgb = pixel_matrix33_multiply_t<PixelR, Matrix, PixelR>( _matrices._LMS_2_RGB )( lms );
		return rgb;
	}
};

template <typename PixelRef, typename PixelR>
const typename pixel_lab_to_rgb_t<PixelRef, PixelR>::MatrixContants pixel_lab_to_rgb_t<PixelRef, PixelR>::_matrices; // init static variable.


}
}
}

#endif


