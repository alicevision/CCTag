#ifndef _TERRY_FILTER_CANNY_HPP_
#define _TERRY_FILTER_CANNY_HPP_

#include "sobel.hpp"
#include "localMaxima.hpp"
#include "floodFill.hpp"
#include "thinning.hpp"

#include <terry/algorithm/transform_pixels.hpp>
#include <terry/algorithm/pixel_by_channel.hpp>
#include <terry/color/norm.hpp>
#include <terry/draw/fill.hpp>
#include <terry/numeric/operations.hpp>
#include <terry/algorithm/pixel_by_channel.hpp>

#include <boost/gil/algorithm.hpp>


namespace terry {
namespace filter {

/**
 * @brief Canny filtering.
 */
template<template<typename> class Alloc, class SView, class TRGBView, class TGrayView, class DView>
void canny(
	const SView& srcView,
	const TRGBView& tmpSobelView,
	const TGrayView& tmpGrayView,
	const DView& cannyView,
	const point2<double>& sobelSize,
	const convolve_boundary_option sobelBoundaryOption,
	const double cannyThresLow, const double cannyThresUpper
	)
{
	typedef typename DView::value_type DPixel;
	typedef typename SView::value_type SPixel;

	typedef typename SView::point_t Point;
	typedef typename channel_mapping_type<DView>::type DChannel;
	typedef typename floating_channel_type_t<DChannel>::type DChannelFloat;
	typedef pixel<DChannelFloat, gray_layout_t> DPixelGray;

	const Point proc_tl( 0, 0 );

	//boost::gil::png_write_view( "data/terry/output_in_terry.png", color_converted_view<rgb8_pixel_t>( srcView ) );
	
	//boost::timer t;
	sobel<Alloc>(
		srcView,
		kth_channel_view<0>(tmpSobelView), // sobel X
		kth_channel_view<1>(tmpSobelView), // sobel Y
		sobelSize,
		sobelBoundaryOption );
	//std::cout << "sobel time: " << t.elapsed() << std::endl;
	
	//t.restart();
	boost::gil::transform_pixels(
		kth_channel_view<0>(tmpSobelView), // srcX
		kth_channel_view<1>(tmpSobelView), // srcY
		kth_channel_view<2>(tmpSobelView), // dst: gradient direction
		algorithm::transform_pixel_by_channel_t<terry::color::channel_norm_t>()
		);
	//std::cout << "norm time: " << t.elapsed() << std::endl;
	
	//boost::gil::png_write_view( "data/terry/output_sobel_terry.png", color_converted_view<rgb8_pixel_t>( tmpSobelView ) );

	//t.restart();
	applyLocalMaxima( tmpSobelView, tmpGrayView );
	//std::cout << "localMaxima time: " << t.elapsed() << std::endl;

	//boost::gil::png_write_view( "data/terry/output_localMaxima_terry.png", color_converted_view<rgb8_pixel_t>( tmpGrayView ) );
	
	//t.restart();
	applyFloodFill<Alloc>( tmpGrayView, cannyView, cannyThresLow, cannyThresUpper );
	//std::cout << "floodFill time: " << t.elapsed() << std::endl;

	//boost::gil::png_write_view( "data/terry/output_floodFill_terry.png", color_converted_view<rgb8_pixel_t>( cannyView ) );

	//t.restart();
	applyThinning( cannyView, tmpGrayView, cannyView );
	//std::cout << "thinning time: " << t.elapsed() << std::endl;
}



}
}

#endif

