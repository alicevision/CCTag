#ifndef _TERRY_ALGORITHM_TRANSFORM_PIXELS_HPP_
#define	_TERRY_ALGORITHM_TRANSFORM_PIXELS_HPP_

#include <terry/math/Rect.hpp>
#include <boost/gil/algorithm.hpp>

namespace terry {
namespace algorithm {

// 1 source/dest without region

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F> GIL_FORCEINLINE
F transform_pixels( const View& dst, F& fun )
{
	for( std::ptrdiff_t y = 0; y < dst.height( ); ++y )
	{
		typename View::x_iterator dstIt = dst.row_begin( y );
		for( std::ptrdiff_t x = 0; x < dst.width( ); ++x )
			fun( dstIt[x] );
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F> GIL_FORCEINLINE
F transform_pixels( const View& dst, const F& fun )
{
	for( std::ptrdiff_t y = 0; y < dst.height( ); ++y )
	{
		typename View::x_iterator dstIt = dst.row_begin( y );
		for( std::ptrdiff_t x = 0; x < dst.width( ); ++x )
			fun( dstIt[x] );
	}
	return fun;
}


////////////////////////////////////////////////////////////////////////////////
///
/// transform_pixels_locator
///
/// \defgroup ImageViewSTLAlgorithmsTransformPixelLocator transform_pixels_locator
/// \ingroup ImageViewSTLAlgorithms
/// \brief for image views (passes locators to the function object, instead of pixel references)
////////////////////////////////////////////////////////////////////////////////

// 1 source/dest without region

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F> GIL_FORCEINLINE
F transform_pixels_locator( const View& dst, F& fun )
{
	typename View::xy_locator dloc = dst.xy_at( 0, 0 );
	for( std::ptrdiff_t y = 0; y < dst.height(); ++y )
	{
		for( std::ptrdiff_t x = 0;
		     x < dst.width();
		     ++x, ++dloc.x() )
		{
			fun( dloc );
		}
		dloc.x() -= dst.width(); ++dloc.y();
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F> GIL_FORCEINLINE
F transform_pixels_locator( const View& dst, const F& fun )
{
	typename View::xy_locator dloc = dst.xy_at( 0, 0 );
	for( std::ptrdiff_t y = 0; y < dst.height(); ++y )
	{
		for( std::ptrdiff_t x = 0;
		     x < dst.width();
		     ++x, ++dloc.x() )
		{
			fun( dloc );
		}
		dloc.x() -= dst.width(); ++dloc.y();
	}
	return fun;
}


// 1 source/dest with regions

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View::xy_locator dloc = dst.xy_at( renderWin.x1-dstRod.x1, renderWin.y1-dstRod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++dloc.x() )
		{
			fun( dloc );
		}
		dloc.x() -= renderWidth; ++dloc.y();
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, const F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View::xy_locator dloc = dst.xy_at( renderWin.x1-dstRod.x1, renderWin.y1-dstRod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++dloc.x() )
		{
			fun( dloc );
		}
		dloc.x() -= renderWidth; ++dloc.y();
	}
	return fun;
}


// 1 source, 1 dest

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View& src, const Rect<std::ptrdiff_t>& srcRod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View::xy_locator sloc = src.xy_at( renderWin.x1-srcRod.x1, renderWin.y1-srcRod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( renderWin.x1-dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++sloc.x(), ++dstIt )
		{
			*dstIt = fun( sloc );
		}
		sloc.x() -= renderWidth; ++sloc.y();
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View& src, const Rect<std::ptrdiff_t>& srcRod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, const F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View::xy_locator sloc = src.xy_at( renderWin.x1-srcRod.x1, renderWin.y1-srcRod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( renderWin.x1-dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++sloc.x(), ++dstIt )
		{
			*dstIt = fun( sloc );
		}
		sloc.x() -= renderWidth; ++sloc.y();
	}
	return fun;
}


// 2 sources, 1 dest

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View1, typename View2, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View1& src1, const Rect<std::ptrdiff_t>& src1Rod,
							const View2& src2, const Rect<std::ptrdiff_t>& src2Rod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View1::xy_locator s1loc = src1.xy_at( renderWin.x1-src1Rod.x1, renderWin.y1-src1Rod.y1 );
	typename View2::xy_locator s2loc = src2.xy_at( renderWin.x1-src2Rod.x1, renderWin.y1-src2Rod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( renderWin.x1-dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++s1loc.x(), ++s2loc.x(), ++dstIt )
		{
			*dstIt = fun( s1loc, s2loc );
		}
		s1loc.x() -= renderWidth; ++s1loc.y();
		s2loc.x() -= renderWidth; ++s2loc.y();
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View1, typename View2, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View1& src1, const Rect<std::ptrdiff_t>& src1Rod,
							const View2& src2, const Rect<std::ptrdiff_t>& src2Rod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, const F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View1::xy_locator s1loc = src1.xy_at( renderWin.x1-src1Rod.x1, renderWin.y1-src1Rod.y1 );
	typename View2::xy_locator s2loc = src2.xy_at( renderWin.x1-src2Rod.x1, renderWin.y1-src2Rod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++s1loc.x(), ++s2loc.x(), ++dstIt )
		{
			*dstIt = fun( s1loc, s2loc );
		}
		s1loc.x() -= renderWidth; ++s1loc.y();
		s2loc.x() -= renderWidth; ++s2loc.y();
	}
	return fun;
}


// 3 sources, 1 dest

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View1, typename View2, typename View3, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View1& src1, const Rect<std::ptrdiff_t>& src1Rod,
							const View2& src2, const Rect<std::ptrdiff_t>& src2Rod,
							const View2& src3, const Rect<std::ptrdiff_t>& src3Rod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View1::xy_locator s1loc = src1.xy_at( renderWin.x1-src1Rod.x1, renderWin.y1-src1Rod.y1 );
	typename View2::xy_locator s2loc = src2.xy_at( renderWin.x1-src2Rod.x1, renderWin.y1-src2Rod.y1 );
	typename View3::xy_locator s3loc = src3.xy_at( renderWin.x1-src3Rod.x1, renderWin.y1-src3Rod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++s1loc.x(), ++s2loc.x(), ++s3loc.x(), ++dstIt )
		{
			*dstIt = fun( s1loc, s2loc, s3loc );
		}
		s1loc.x() -= renderWidth; ++s1loc.y();
		s2loc.x() -= renderWidth; ++s2loc.y();
		s3loc.x() -= renderWidth; ++s3loc.y();
	}
	return fun;
}

/// \ingroup ImageViewSTLAlgorithmsTransformPixels
/// \brief std::transform for image views
template <typename View1, typename View2, typename View3, typename ViewDst, typename F>
GIL_FORCEINLINE
F transform_pixels_locator( const View1& src1, const Rect<std::ptrdiff_t>& src1Rod,
							const View2& src2, const Rect<std::ptrdiff_t>& src2Rod,
							const View2& src3, const Rect<std::ptrdiff_t>& src3Rod,
							const ViewDst& dst, const Rect<std::ptrdiff_t>& dstRod,
							const Rect<std::ptrdiff_t>& renderWin, const F& fun )
{
	const std::ptrdiff_t renderWidth = renderWin.x2 - renderWin.x1;
	typename View1::xy_locator s1loc = src1.xy_at( renderWin.x1-src1Rod.x1, renderWin.y1-src1Rod.y1 );
	typename View2::xy_locator s2loc = src2.xy_at( renderWin.x1-src2Rod.x1, renderWin.y1-src2Rod.y1 );
	typename View3::xy_locator s3loc = src3.xy_at( renderWin.x1-src3Rod.x1, renderWin.y1-src3Rod.y1 );
	for( std::ptrdiff_t y = renderWin.y1; y < renderWin.y2; ++y )
	{
		typename ViewDst::x_iterator dstIt = dst.x_at( dstRod.x1, y-dstRod.y1 );
		for( std::ptrdiff_t x = renderWin.x1;
		     x < renderWin.x2;
		     ++x, ++s1loc.x(), ++s2loc.x(), ++s3loc.x(), ++dstIt )
		{
			*dstIt = fun( s1loc, s2loc, s3loc );
		}
		s1loc.x() -= renderWidth; ++s1loc.y();
		s2loc.x() -= renderWidth; ++s2loc.y();
		s3loc.x() -= renderWidth; ++s3loc.y();
	}
	return fun;
}


}
}

#endif

