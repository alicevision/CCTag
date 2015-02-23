#include "canny.hpp"

#ifdef ROM_USE_TUTTLE
#include <boost/preprocessor/stringize.hpp>
#include <tuttle/host/InputBufferNode.hpp>
#endif

namespace rom {
namespace vision {
namespace marker {
namespace cctag {

/**
 * @param[out] cannyView output gray view
 * @param[in] srcImage gray source image
 */
void cvCanny( boost::gil::rgb32f_view_t & cannyView, const IplImage* srcImage, const double thrLow, const double thrHigh )
{
	using namespace boost::gil;
	
	typedef boost::gil::rgb32f_view_t CannyView;
	typedef CannyView::value_type CannyPixel;
	typedef channel_type<CannyPixel>::type CannyChannel;
	
	BOOST_ASSERT( cannyView.width() == srcImage->width );
	BOOST_ASSERT( cannyView.height() == srcImage->height );
	

	/**************************************************************************
	 *  apply canny edges detection algorithm                                *
	 **************************************************************************/
	// Compute canny
	try
	{
            
                // TODO: Use global allocation (memory pool) or hack with static variable.
		//gray16_image_t<poolAllocator> dxImg( cannyView.width(), cannyView.height() );
		CvMat* dx = NULL;
		CvMat* dy = NULL;
		// opencv works only on gray 8bits images...
                // TODO: no local image allocation without using memory pool.
		gray8_image_t cannyImgBuffer( cannyView.width(), cannyView.height() );
		gray8_view_t cannyViewBuffer( view(cannyImgBuffer) );
		boostCv::CvImageView cannyImg( cannyViewBuffer );

		cvRecodedCanny( const_cast<IplImage*>(srcImage), cannyImg.get(), dx, dy, thrLow * 256, thrHigh * 256, /*7*/ 3 | CV_CANNY_L2_GRADIENT );
		//cvRecodedCannyGPUFilter2D( simg, cannyImg, dx, dy, thrLow * 256, thrHigh * 256, 7 | CV_CANNY_L2_GRADIENT );
		BOOST_ASSERT( dx && dy );

                // imwrite( "output.tiff", dx );
                
		boost::int16_t *pdx = dx->data.s;
		boost::int16_t *pdy = dy->data.s;
//		float *pdx = dx->data.fl;
//		float *pdy = dy->data.fl;
		for( int y = 0; y < cannyView.height(); ++y )
		{
			CannyView::x_iterator it = cannyView.row_begin( y );
			gray8_view_t::x_iterator itcv = cannyViewBuffer.row_begin( y );
			for( int x = 0; x < cannyView.width(); ++x )
			{
				(*it)[0] = channel_convert<CannyChannel, boost::gil::bits8>( (*itcv)[0] );
				(*it)[1] = *pdx;
				(*it)[2] = *pdy;
				++it;
				++pdx;
				++pdy;
				++itcv;
			}
		}

		cvReleaseMat( &dx );
		cvReleaseMat( &dy );
	}
	catch( std::exception & e )
	{
		std::cerr << e.what() << std::endl;
	}
}

}
}
}
}


