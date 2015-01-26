#include "canny.hpp"

#ifdef ROM_USE_TUTTLE
#include <boost/preprocessor/stringize.hpp>
#include <tuttle/host/InputBufferNode.hpp>
#endif

namespace rom {
namespace vision {
namespace marker {
namespace cctag {

#ifdef ROM_USE_TUTTLE
void createCannyGraph( tuttle::host::Graph & canny, tuttle::host::InputBufferNode* & cannyInputBuffer, tuttle::host::Graph::Node* & cannyOutput, tuttle::host::Graph::Node* & sobelOutput )
{
	using namespace tuttle::host;

	try
	{
	InputBufferNode& inputBuffer1 = canny.createInputBuffer();
//	Graph::Node& bitdepth1    = canny.createNode( "fr.tuttle.bitdepth" );
	Graph::Node& bitdepth2    = canny.createNode( "fr.tuttle.bitdepth" );
	Graph::Node& blur1        = canny.createNode( "fr.tuttle.blur" );
	Graph::Node& blur2        = canny.createNode( "fr.tuttle.blur" );
	Graph::Node& sobel1       = canny.createNode( "fr.tuttle.duranduboi.sobel" );
	Graph::Node& sobel2       = canny.createNode( "fr.tuttle.duranduboi.sobel" );
	Graph::Node& localMaxima  = canny.createNode( "fr.tuttle.duranduboi.localmaxima" );
	Graph::Node& floodfill    = canny.createNode( "fr.tuttle.duranduboi.floodfill" );
	Graph::Node& thinning     = canny.createNode( "fr.tuttle.duranduboi.thinning" );
//		Graph::Node& write0       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write1a       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write1b       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write2       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write3       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write4       = canny.createNode( "fr.tuttle.pngwriter" );
//		Graph::Node& write5       = canny.createNode( "fr.tuttle.pngwriter" );

//		inputBuffer1.setClipBitDepth( InputBufferNode::eBitDepthUByte );
	inputBuffer1.setClipBitDepth( InputBufferNode::eBitDepthFloat );
	inputBuffer1.setClipComponent( InputBufferNode::ePixelComponentAlpha );
	//inputBuffer1.setClipRawBuffer( /*static_cast<char*>*/(char*)(boost::gil::interleaved_view_get_raw_data( imgView )) );

	cannyInputBuffer = &inputBuffer1;
	cannyOutput = &bitdepth2;
	sobelOutput = &sobel2;

	static const double kernelEpsilon = 0.1;

//	bitdepth1.getParam( "outputBitDepth" ).set( 3 );
	bitdepth2.getParam( "outputBitDepth" ).set( 1 );

	blur1.getParam( "border" ).set( "Mirror" );
	blur1.getParam( "size" ).set( 1.0, 0.0 );
	blur1.getParam( "normalizedKernel" ).set( false );
	blur1.getParam( "kernelEpsilon" ).set( kernelEpsilon );

	blur2.getParam( "border" ).set( "Mirror" );
	blur2.getParam( "size" ).set( 0.0, 1.0 );
	blur2.getParam( "normalizedKernel" ).set( false );
	blur2.getParam( "kernelEpsilon" ).set( kernelEpsilon );

	sobel1.getParam( "border" ).set( "Mirror" );
	sobel1.getParam( "size" ).set( 1.0, 1.0 );
	sobel1.getParam( "normalizedKernel" ).set( false );
	sobel1.getParam( "computeGradientDirection" ).set( false );
	sobel1.getParam( "kernelEpsilon" ).set( kernelEpsilon );
	sobel1.getParam( "pass" ).set( 1 );
	sobel1.getParam( "outputComponent" ).set( "RGBA" );

	sobel2.getParam( "border" ).set( "Mirror" );
	sobel2.getParam( "size" ).set( 1.0, 1.0 );
	sobel2.getParam( "normalizedKernel" ).set( false );
	sobel2.getParam( "computeGradientDirection" ).set( false );
	sobel2.getParam( "kernelEpsilon" ).set( kernelEpsilon );
	sobel2.getParam( "pass" ).set( 2 );
	sobel2.getParam( "outputComponent" ).set( "RGBA" );

	localMaxima.getParam( "outputComponent" ).set( "Alpha" );

	floodfill.getParam( "upperThres" ).set( 0.1 );
	floodfill.getParam( "lowerThres" ).set( 0.025 );

	thinning.getParam( "border" ).set( "Black" );

//		write0.getParam( "components" ).set( "rgba" );
//		write1a.getParam( "components" ).set( "rgba" );
//		write1b.getParam( "components" ).set( "rgba" );
//		write2.getParam( "components" ).set( "rgba" );
//		write2.getParam( "components" ).set( "rgba" );
//		write3.getParam( "components" ).set( "rgba" );
//		write5.getParam( "components" ).set( "rgba" );

//		write5.getParam( "filename" ).set( "data/canny/5_finalBitDepth.png" );
//		write0.getParam( "filename" ).set( "data/canny/0_blur.png" );
//		write1a.getParam( "filename" ).set( "data/canny/1a_sobel.png" );
//		write1b.getParam( "filename" ).set( "data/canny/1b_sobel.png" );
//		write2.getParam( "filename" ).set( "data/canny/2_localMaxima.png" );
//		write3.getParam( "filename" ).set( "data/canny/3_floodfill.png" );
//		write4.getParam( "filename" ).set( "data/canny/4_thinning.png" );

	canny.connect( inputBuffer1, blur1 );
	canny.connect( blur1, blur2 );
	canny.connect( blur2, sobel1 );
	canny.connect( sobel1, sobel2 );
	canny.connect( sobel2, localMaxima );
	canny.connect( localMaxima, floodfill );
	canny.connect( floodfill, thinning );
	canny.connect( thinning, bitdepth2 );

//		_canny.connect( blur2, write0 );
//		_canny.connect( sobel1, write1a );
//		_canny.connect( sobel2, write1b );
//		_canny.connect( localMaxima, write2 );
//		_canny.connect( floodfill, write3 );
//		_canny.connect( thinning, write4 );
//		_canny.connect( bitdepth2, write5 );

//	_cannyOutputs.push_back( write0.getName() );
//	_cannyOutputs.push_back( write1a.getName() );
//	_cannyOutputs.push_back( write1b.getName() );
//	_cannyOutputs.push_back( write2.getName() );
//	_cannyOutputs.push_back( write3.getName() );
//	_cannyOutputs.push_back( write4.getName() );
//	_cannyOutputs.push_back( write5.getName() );
	}
	catch(boost::exception & e)
	{
		std::cerr << "error!" << boost::diagnostic_information( e );
	}
}

#else

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

#endif

}
}
}
}


