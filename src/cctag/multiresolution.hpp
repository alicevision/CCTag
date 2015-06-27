#ifndef VISION_CCTAG_MULTIRESOLUTION_HPP_
#define VISION_CCTAG_MULTIRESOLUTION_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/params.hpp>
#include <cctag/frame.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Cercle.hpp>
#include <cctag/boostCv/cvImage.hpp>
#include <cctag/ImagePyramid.hpp>

#include <terry/sampler/all.hpp>
#include <terry/sampler/resample_subimage.hpp>

#include <boost/gil/image.hpp>

#include <cstddef>
#include <cmath>
#include <vector>

//#define USE_RESIZE_OCV3

namespace cctag {

struct CCTagParams
{
};

/**
 * @brief Detect all CCTag in the image using multiresolution detection.
 * 
 * @param[out] markers detected cctags
 * @param[in] srcImg
 * @param[in] frame
 * 
 */

void cctagMultiresDetection(
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid & imagePyramid,
        const std::size_t frame,
        const Parameters & params);

void update(CCTag::List& markers, const CCTag& markerToAdd);

/**
 * @todo to put outside !!!
 */
template <class View>
struct image_from_view
{
        typedef typename View::value_type value_type; // pixel_t
        typedef typename boost::gil::image<value_type, boost::gil::is_planar<View>::value> type;
};

template<class View>
class PyramidImage
{
public:
	typedef typename image_from_view<View>::type Image;

	PyramidImage( const View& srcImg, const std::size_t nbLevels )
	{
		buildPyramidFromView( srcImg, nbLevels );
	}
	PyramidImage( const std::size_t rootWidth, const std::size_t rootHeight, const std::size_t nbLevels )
	{
		buildPyramid( rootWidth, rootHeight, nbLevels );
	}
	View getView( const std::size_t level )
	{
		return _pyramidViews[level];
	}
private:
	void buildPyramid( const std::size_t rootWidth, const std::size_t rootHeight, const std::size_t nbLevels )
	{
		_pyramidImages.resize( nbLevels+1 );
		_pyramidViews.resize( nbLevels+1 );
		for( std::size_t level = 0; level <= nbLevels; ++level )
		{
			const double inv_scale = std::pow(2.0, (int)level);
			const double scale = 1.0 / inv_scale;
			_pyramidImages[level].recreate( scale * rootWidth, scale * rootHeight );
			_pyramidViews[level] = boost::gil::view( _pyramidImages[level] );
		}
		_srcView = _pyramidViews[0];
	}
	void buildPyramidFromView( const View& srcView, const std::size_t nbLevels )
	{
		_srcView = srcView;
		_pyramidImages.resize( nbLevels );
		_pyramidViews.resize( nbLevels+1 );
		_pyramidViews[0] = srcView;
		for( std::size_t level = 0; level < nbLevels; ++level )
		{
			const double inv_scale = std::pow(2.0, (int)level+1);
			const double scale = 1.0 / inv_scale;
			buildImageAtScale( _pyramidImages[level], srcView, scale );
			_pyramidViews[level+1] = boost::gil::view( _pyramidImages[level] );
		}
	}
	void buildImageAtScale( Image& image, const View& srcView, const double scale )
	{
		using namespace boost::gil;
                image.recreate( scale * srcView.width(), scale * srcView.height() );
#ifndef USE_RESIZE_OCV3
                  terry::resize_view( srcView, boost::gil::view( image ), terry::sampler::bilinear_sampler() );
#else
                  boostCv::CvImageView cvviewSrc(srcView);
                  IplImage * img = cvviewSrc.get();

                  boostCv::CvImageView cvviewDst(boost::gil::view( image ));
                  cvResize(img, cvviewDst.get());
#endif
	}
private:
	View _srcView; // if we use an external source view (for root image)
	std::vector<Image> _pyramidImages;
	std::vector<View> _pyramidViews;
};

/**
 * @brief Create a pyramid image with \p nbLevels levels.
 * 
 * @param[out] multires the output pyramidal image.
 * @param[in] srcImg input image (full resolution).
 * @param[in] nbLevels number of levels in the pyramid image (0 do nothing).
 */
template<class View>
void createMultiResolutionImage( PyramidImage<View>& multires, const View& srcImg, const std::size_t nbLevels );

void clearDetectedMarkers( const std::map<std::size_t, CCTag::List> & pyramidMarkers, const boost::gil::rgb32f_view_t & cannyRGB, const std::size_t curLevel );

} // namespace cctag


#endif

