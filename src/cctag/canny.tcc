#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/filter/cvRecode.hpp>

#include <cctag/boostCv/cvImage.hpp>

#include <terry/filter/thinning.hpp>
#include <terry/sampler/sampler.hpp>
#include <terry/numeric/init.hpp>

#include <opencv/cv.h>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/gil/image_view_factory.hpp>

namespace rom {
namespace vision {
namespace marker {
namespace cctag {


#ifdef ROM_USE_TUTTLE

template<class SView, class CannyView, class GradXView, class GradYView>
void cannyTuttle( std::vector<memory::CACHE_ELEMENT>& datas, const SView& svw, CannyView& cannyView, GradXView& cannyGradX, GradYView& cannyGradY, const double thrCannyLow, const double thrCannyHigh )
{
	ROM_COUT_DEBUG( "USING TUTTLE CANNY" );
	using namespace boost::gil;
	using namespace tuttle::host;
	gray32f_image_t fimg( svw.dimensions() );
	gray32f_view_t fsvw( view( fimg ) );
	copy_and_convert_pixels( svw, fsvw );

	TUTTLE_COUT_INFOS;

	tuttle::host::Graph canny;
	tuttle::host::InputBufferNode* cannyInputBuffer;
	std::list<std::string> cannyOutputs;
	tuttle::host::Graph::Node* cannyOutput;
	tuttle::host::Graph::Node* sobelOutput;

	createCannyGraph( canny, cannyInputBuffer, cannyOutput, sobelOutput );

	cannyOutputs.push_back( cannyOutput->getName() );
	cannyOutputs.push_back( sobelOutput->getName() );

	OfxRectD ibRod = { 0, 0, fsvw.width(), fsvw.height() };
	cannyInputBuffer->setClipRod( ibRod );
	cannyInputBuffer->setClipRawBuffer( (char*)interleaved_view_get_raw_data( fsvw ) );

	boost::posix_time::ptime t1a( boost::posix_time::microsec_clock::local_time() );
	ROM_COUT_DEBUG( "Computing frame: " << frame );
	memory::MemoryCache res0 = canny.compute( cannyOutputs, frame );
	TUTTLE_COUT_INFOS;
	boost::posix_time::ptime t2a( boost::posix_time::microsec_clock::local_time() );

	ROM_COUT_DEBUG( "Process tuttle canny took: " << t2a - t1a );

	TUTTLE_COUT_INFOS;
	memory::CACHE_ELEMENT cannyRes = res0.get( cannyOutput->getName(), frame );
	TUTTLE_COUT_INFOS;
	memory::CACHE_ELEMENT sobelRes = res0.get( sobelOutput->getName(), frame );

	TUTTLE_COUT_INFOS;
	TUTTLE_COUT_VAR( cannyRes->getROD() );
	TUTTLE_COUT_VAR( cannyRes->getBounds() );
	TUTTLE_COUT_VAR( cannyRes->getComponentsType() );
	TUTTLE_COUT_VAR( cannyRes->getBitDepth() );
	TUTTLE_COUT_INFOS;
	cannyView = cannyRes->getGilView<boost::gil::gray8_view_t>();

	rgba32f_view_t sobelView = sobelRes->getGilView<boost::gil::rgba32f_view_t>();

	cannyGradX = kth_channel_view<0>( sobelView );
	cannyGradY = kth_channel_view<1>( sobelView );

//		png_write_view( "sobelX.png", boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>( cannyGradX ) );
//		png_write_view( "sobelY.png", boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>( cannyGradY ) );

	TUTTLE_COUT_INFOS;
	datas.push_back( cannyRes );
	datas.push_back( sobelRes );
}

#else

template<class SView, class CannyRGBView, class CannyView, class GradXView, class GradYView>
void cannyCv( const SView& srcView, CannyRGBView& cannyRGB, CannyView& cannyView, GradXView& cannyGradX, GradYView& cannyGradY, const double thrCannyLow, const double thrCannyHigh )
{
	using namespace boost::gil;
	ROM_COUT_DEBUG( "USING CV CANNY" );

	typedef pixel<typename channel_type<SView>::type, layout<gray_t> > PixelGray;
	typedef image<PixelGray, false> GrayImage;
	typedef typename GrayImage::view_t GrayView;
	typedef typename channel_type<GrayView>::type Precision;

	if ( num_channels<SView>::type::value != 1 )
	{
		GrayImage gimg( srcView.dimensions() );
		GrayView graySrcView( view( gimg ) );
		copy_and_convert_pixels( srcView, graySrcView );
		// Apply canny
//			boost::posix_time::ptime t1a( boost::posix_time::microsec_clock::local_time() );
		cctag::cvCanny( cannyRGB, boostCv::CvImageView( graySrcView ).get(), thrCannyLow, thrCannyHigh );
//			rom::graphics::cuda::canny( gsvw, cannyRGB, thrCannyLow, thrCannyHigh );
//			boost::posix_time::ptime t2a( boost::posix_time::microsec_clock::local_time() );
//			ROM_TCOUT( "Process cvCanny took: " << t2a - t1a );
	}
	else
	{
		boost::posix_time::ptime t1a( boost::posix_time::microsec_clock::local_time() );
		cctag::cvCanny( cannyRGB, boostCv::CvImageView( srcView ).get(), thrCannyLow, thrCannyHigh );
//			rom::graphics::cuda::canny( svw, cannyRGB, thrCannyLow, thrCannyHigh );
		boost::posix_time::ptime t2a( boost::posix_time::microsec_clock::local_time() );
//			ROM_TCOUT( "Process cvCanny took: " << t2a - t1a );
	}
	// Thinning
	using namespace boost::gil;
	using namespace terry;
	using namespace terry::numeric;
	typedef Rect<std::ptrdiff_t> rect_t;
	rect_t srcRod( 0, 0, cannyView.width(), cannyView.height() );
	rect_t srcRodCrop1( 1, 1, cannyView.width() - 1, cannyView.height() - 1 );
	rect_t srcRodCrop2( 1, 1, cannyView.width() - 1, cannyView.height() - 1 );
	rect_t procWindowRoWCrop1 = srcRodCrop1;
	rect_t procWindowRoWCrop2 = srcRodCrop2;
	rom::IPoolDataPtr dataTmp = rom::MemoryPool::instance().allocate( cannyView.width() * cannyView.height() * sizeof(Precision) );
	GrayView view_tmp = interleaved_view( srcView.width(), srcView.height(), (PixelGray*)dataTmp->data(), cannyView.width() * sizeof(Precision) );
	PixelGray pixelZero;
	pixel_zeros_t<PixelGray>()( pixelZero );
	fill_pixels( view_tmp, pixelZero );
	algorithm::transform_pixels_locator( cannyView, srcRod,
							  view_tmp, srcRod,
							  procWindowRoWCrop1,
							  terry::filter::thinning::pixel_locator_thinning_t<CannyView, GrayView>( cannyView, terry::filter::thinning::lutthin1 ) );

	fill_pixels( cannyView, pixelZero );
	algorithm::transform_pixels_locator( view_tmp, srcRod,
							  cannyView, srcRod,
							  procWindowRoWCrop2,
							  terry::filter::thinning::pixel_locator_thinning_t<GrayView, CannyView>( view_tmp, terry::filter::thinning::lutthin2 ) );

//		png_write_view( "sobelX.png", color_converted_view<gray8_pixel_t>( cannyGradX ) );
//		png_write_view( "sobelY.png", color_converted_view<gray8_pixel_t>( cannyGradY ) );
//		png_write_view( "canny.png", color_converted_view<gray8_pixel_t>( cannyView ) );

/*
	#ifdef USER_LILIAN
	{
		rgb8_image_t tmpI( cannyView.dimensions() );
		rgb8_view_t tmpV( view( tmpI ) );
		copy_and_convert_pixels( cannyView, tmpV );
		std::ostringstream tmps;
		tmps << "/home/lcalvet/cpp_workspace/rom/data/output/canny/canny_" << frame << ".png";
		png_write_view( tmps.str(), tmpV );
	}
	#endif
*/
}

#endif



template<class CView, class DXView, class DYView>
void edgesPointsFromCanny( std::vector<EdgePoint>& points, EdgePointsImage & edgePointsMap, CView & cannyView, DXView & dx, DYView & dy )
{
	using namespace boost::gil;

	edgePointsMap.resize( boost::extents[cannyView.width()][cannyView.height()] );
	std::fill( edgePointsMap.origin(), edgePointsMap.origin() + edgePointsMap.size(), (EdgePoint*)NULL );

	points.reserve( cannyView.width() * cannyView.height() / 2 ); //TODO évaluer la borne max du nombre de point contour

	for( int y = 0 ; y < cannyView.height(); ++y )
	{
		typename CView::x_iterator itc = cannyView.row_begin( y );
		typename DXView::x_iterator itDx = dx.row_begin( y );
		typename DYView::x_iterator itDy = dy.row_begin( y );
		for( int x = 0 ; x < cannyView.width(); ++x )
		{
			if ( (*itc)[0] )
			{
				points.push_back( EdgePoint( x, y, (*itDx)[0], (*itDy)[0] ) );
				EdgePoint* p = &points.back();
				edgePointsMap[x][y] = p;
			}
			++itc;
			++itDx;
			++itDy;
		}
	}

}

}
}
}
}

