#ifndef _TERRY_COLOR_LAYOUT_YUV_HPP_
#define	_TERRY_COLOR_LAYOUT_YUV_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

/// \addtogroup ColorNameModel
/// \{
namespace yuv
{
/// \brief Luminance
struct y_t {};
/// \brief U
struct u_t {};
/// \brief V
struct v_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3<
		yuv::y_t,
		yuv::u_t,
		yuv::v_t
	> yuv_t;

/**
 * @brief RGB -> YUV
 */
template < typename SrcP, typename DstP >
void layout_convert<rgb_t, yuv_t>( const SrcP& src, DstP& dst )
{
	//std::cout << "convert RGB to YUV" << std::endl;
	get_color( dst, yuv::y_t() )	= get_color( src, red_t() )                                         + 1.13983 * get_color( src, blue_t() );
	get_color( dst, yuv::u_t() )	= get_color( src, red_t() ) - 0.39465 * get_color( src, green_t() ) - 0.58060 * get_color( src, blue_t() );
	get_color( dst, yuv::v_t() )	= get_color( src, red_t() ) + 2.03211 * get_color( src, green_t() )                                       ;
}

/**
 * @brief YUV -> RGB
 */
template < typename SrcP, typename DstP >
void layout_convert<yuv_t, rgb_t>( const SrcP& src, DstP& dst )
{
	//std::cout << "convert YUV to RGB" << std::endl;
	get_color( dst, red_t() )	=  0.299   * get_color( src, yuv::y_t() ) + 0.587   * get_color( src, yuv::u_t() ) + 0.114   * get_color( src, yuv::v_t() );
	get_color( dst, green_t() )	= -0.14713 * get_color( src, yuv::y_t() ) - 0.28886 * get_color( src, yuv::u_t() ) + 0.436   * get_color( src, yuv::v_t() );
	get_color( dst, blue_t() )	=  0.615   * get_color( src, yuv::y_t() ) - 0.51499 * get_color( src, yuv::u_t() ) - 0.10001 * get_color( src, yuv::v_t() );
}


}
}
}


#endif
