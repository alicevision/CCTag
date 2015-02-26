#ifndef _CCTAG_IMAGE_HPP_
#define	_CCTAG_IMAGE_HPP_

namespace rom {
namespace img {

template<class SView>
inline boost::gil::gray8_view_t toGray( const SView & sView, boost::gil::gray8_image_t& grayImg )
{
	using namespace boost::gil;
	grayImg.recreate( sView.dimensions() );
	gray8_view_t grayView = view( grayImg );
	copy_and_convert_pixels( sView, grayView );
	return grayView;
}

template<>
inline boost::gil::gray8_view_t toGray( const boost::gil::gray8_view_t & sView, boost::gil::gray8_image_t& grayImg )
{
	return sView;
}

}
}

#endif	/* IMAGE_HPP */

