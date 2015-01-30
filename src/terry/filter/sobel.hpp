#ifndef _TERRY_FILTER_SOBEL_HPP_
#define	_TERRY_FILTER_SOBEL_HPP_

#include "gaussianKernel.hpp"

#include <terry/filter/convolve.hpp>
#include <terry/numeric/operations.hpp>

namespace terry {
namespace filter {

/**
 * @brief Sobel filtering.
 */
template<template<typename> class Alloc, class SView, class DView>
void sobel( const SView& srcView, const DView& dstViewX, const DView& dstViewY, const point2<double>& size, const convolve_boundary_option boundary_option )
{
	typedef typename SView::point_t Point;
	typedef typename channel_mapping_type<DView>::type DChannel;
	typedef typename floating_channel_type_t<DChannel>::type DChannelFloat;
	typedef pixel<DChannelFloat, gray_layout_t> DPixelGray;

	const bool normalizedKernel = false;
	const double kernelEpsilon = 0.001;
	const Point proc_tl( 0, 0 );

	typedef float Scalar;
	kernel_1d<Scalar> xKernelGaussianDerivative = buildGaussianDerivative1DKernel<Scalar>( size.x, normalizedKernel, kernelEpsilon );
	kernel_1d<Scalar> xKernelGaussian = buildGaussian1DKernel<Scalar>( size.x, normalizedKernel, kernelEpsilon );
	kernel_1d<Scalar> yKernelGaussianDerivative = buildGaussianDerivative1DKernel<Scalar>( size.y, normalizedKernel, kernelEpsilon );
	kernel_1d<Scalar> yKernelGaussian = buildGaussian1DKernel<Scalar>( size.y, normalizedKernel, kernelEpsilon );
	
	correlate_rows_cols_auto<DPixelGray, Alloc>(
		color_converted_view<DPixelGray>( srcView ),
		xKernelGaussianDerivative,
		xKernelGaussian,
		dstViewX,
		proc_tl,
		boundary_option );
	
	correlate_rows_cols_auto<DPixelGray, Alloc>(
		color_converted_view<DPixelGray>( srcView ),
		yKernelGaussian,
		yKernelGaussianDerivative,
		dstViewY,
		proc_tl,
		boundary_option );
}

}
}

#endif

