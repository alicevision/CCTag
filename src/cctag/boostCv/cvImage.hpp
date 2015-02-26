#ifndef _CCTAG_CVIMAGE_HPP_
#define	_CCTAG_CVIMAGE_HPP_

#include <cctag/progBase/exceptions.hpp>
#include <cctag/global.hpp>

#include <boost/gil/gil_all.hpp>
#include <boost/type_traits/is_signed.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

namespace rom {
namespace boostCv {

/**
 * @brief This function only creates a cvImage which points to the sView buffer.
 *
 * @warning release with releaseCvImageView, and NOT directly with cvReleaseImage !
 */
template<class View>
IplImage* createCvImageHeader( const View & sView )
{
	using namespace boost::gil;
	typedef typename channel_type<View>::type ChannelType;

	static const int depth = sizeof(ChannelType) * 8 | (boost::is_signed<ChannelType>::value?IPL_DEPTH_SIGN:0);

	IplImage* img = cvCreateImageHeader( cvSize( sView.width(), sView.height() ),
										 depth,
										 num_channels<View>::value );
	if( !img )
	{
		BOOST_THROW_EXCEPTION( exception::BadAlloc() );
	}
	img->imageData = (char*)( interleaved_view_get_raw_data( sView ) );
	img->widthStep = sView.pixels().row_size();
	return img;
}

template<class View>
IplImage* createCvImage( const std::size_t width, const std::size_t height )
{
	typedef typename boost::gil::channel_type<View>::type ChannelType;
	static const int depth = sizeof(ChannelType) * 8 | (boost::is_signed<ChannelType>::value?IPL_DEPTH_SIGN:0);
	IplImage* img = cvCreateImage( cvSize( width, height ),
									 depth,
									 boost::gil::num_channels<View>::value
									 );
	return img;
}

inline void releaseCvImageHeader( IplImage*& img )
{
	img->imageData = NULL;
	cvReleaseImage( &img );
}

class CvImageView
{
public:
	CvImageView()
	: _img(NULL)
	{}
	CvImageView( IplImage* const img )
	: _img(img)
	{}
	/// @todo not allowed to copy a CvImageView
	CvImageView( CvImageView& other );
	/// @brief not allowed to get the pointer of a const CvImageView
	CvImageView( const CvImageView& other );
	template<class View>
	CvImageView( const View & sView )
	{
		_img = createCvImageHeader( sView );
	}
	~CvImageView()
	{
		reset();
	}

	CvImageView& operator=( IplImage* const img )
	{
		reset( img );
		return *this;
	}
	void reset( IplImage* const img = NULL )
	{
		if( _img )
			releaseCvImageHeader( _img );
		_img = img;
	}

	const IplImage* const get() const { return _img; }
	      IplImage*       get()       { return _img; }

	const IplImage* const operator->() const { return _img; }
	      IplImage*       operator->()       { return _img; }

	IplImage* release() { IplImage* m = _img; _img = NULL; return m; }

private:
	IplImage* _img;
};

class CvImageContainer
{
public:
	CvImageContainer()
	: _img(NULL)
	{}
	CvImageContainer( IplImage* const img )
	: _img(img)
	{}
	/// @todo not allowed to copy a CvImageContainer
	CvImageContainer( CvImageContainer& other );
	/// @brief not allowed to get the pointer of a const CvImageContainer
	CvImageContainer( const CvImageContainer& other );
	~CvImageContainer()
	{
		reset();
	}

	CvImageContainer& operator=( IplImage* img )
	{
		reset( img );
		return *this;
	}
	void reset( IplImage* img = NULL )
	{
		if( _img )
			cvReleaseImage( &_img );
		_img = img;
	}

	const IplImage* const get() const { return _img; }
	      IplImage*       get()       { return _img; }

	const IplImage* const operator->() const { return _img; }
	      IplImage*       operator->()       { return _img; }

	IplImage* release() { IplImage* m = _img; _img = NULL; return m; }

private:
	mutable IplImage* _img;
};

}
}


inline std::ostream& operator<<( std::ostream& os, const IplImage & image )
{
	os << "Image " << image.width << "x" << image.height << " depth:" << image.depth << " channels:" << image.nChannels << std::endl;
	return os;
}

inline std::ostream& operator<<( std::ostream& os, const rom::boostCv::CvImageView & image )
{
	os << *image.get();
	return os;
}

inline std::ostream& operator<<( std::ostream& os, const rom::boostCv::CvImageContainer & image )
{
	os << *image.get();
	return os;
}


#endif
