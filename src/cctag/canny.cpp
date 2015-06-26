#include <cctag/canny.hpp>

#include <boost/gil/image_view.hpp>

//#define USE_CANNY_OCV3
#ifdef USE_CANNY_OCV3
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/timer.hpp>
#endif

#ifdef CCTAG_USE_TUTTLE
#include <boost/preprocessor/stringize.hpp>
#include <tuttle/host/InputBufferNode.hpp>
#endif

namespace cctag
{

/**
 * @param[out] cannyView output gray view
 * @param[in] srcImage gray source image
 */
void cvCanny(
        const cv::Mat & imgGraySrc,
        cv::Mat & imgCanny,
        cv::Mat & imgDX,
        cv::Mat & imgDY,
        cv::Mat & imgMag,
        const double thrLow,
        const double thrHigh )
{
  using namespace boost::gil;

  //typedef boost::gil::rgb32f_view_t CannyView;
  //typedef CannyView::value_type CannyPixel;
  //typedef channel_type<CannyPixel>::type CannyChannel;

  //BOOST_ASSERT( cannyView.width() == srcImage->width );
  //BOOST_ASSERT( cannyView.height() == srcImage->height );

  // Compute canny
  try
  {
    // TODO: Use global allocation (memory pool) or hack with static variable.
    //gray16_image_t<poolAllocator> dxImg( cannyView.width(), cannyView.height() );
    
//    CvMat* dx = NULL;
//    CvMat* dy = NULL;
    
    // opencv works only on gray 8bits images...
    // TODO: no local image allocation without using memory pool.
    
//    gray8_image_t cannyImgBuffer( cannyView.width(), cannyView.height() );
//    gray8_view_t cannyViewBuffer( view(cannyImgBuffer) );
//    boostCv::CvImageView cannyImg( cannyViewBuffer );

    //boost::timer t;
    
    cvRecodedCanny( imgGraySrc, imgCanny, imgDX, imgDY, thrLow * 256, thrHigh * 256, /*7*/ 3 | CV_CANNY_L2_GRADIENT );
    //CCTAG_COUT( "Time for cvRecodedCanny " << t.elapsed() );
    
#ifdef USE_CANNY_OCV3
	cv::Mat matSrc(cv::cvarrToMat( const_cast<IplImage*>(srcImage) ));
	cv::Mat matCanny(cv::cvarrToMat( const_cast<IplImage*>(cannyImg.get()) ));
	t.restart();
	cv::Canny( matSrc, matCanny, thrLow * 256, thrHigh * 256, 7 );
	CCTAG_COUT( "Time for cv::Canny " << t.elapsed() );
	//cv::imwrite("/home/lilian/data/toto.png",matCanny);
#endif
    
//    BOOST_ASSERT( dx && dy );
//
//    boost::int16_t *pdx = dx->data.s;
//    boost::int16_t *pdy = dy->data.s;
//
//    for( int y = 0; y < cannyView.height(); ++y )
//    {
//      CannyView::x_iterator it = cannyView.row_begin( y );
//      gray8_view_t::x_iterator itcv = cannyViewBuffer.row_begin( y );
//      for( int x = 0; x < cannyView.width(); ++x )
//      {
//        (*it)[0] = channel_convert<CannyChannel, boost::gil::bits8>( (*itcv)[0] );
//        (*it)[1] = *pdx;
//        (*it)[2] = *pdy;
//        ++it;
//        ++pdx;
//        ++pdy;
//        ++itcv;
//      }
//    }
//
//    cvReleaseMat( &dx );
//    cvReleaseMat( &dy );
  }
  catch( std::exception & e )
  {
    std::cerr << e.what() << std::endl;
  }
}

} // namespace cctag


