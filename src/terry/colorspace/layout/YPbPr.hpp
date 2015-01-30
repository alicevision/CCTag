#ifndef _TERRY_COLOR_LAYOUT_YPBPR_HPP_
#define	_TERRY_COLOR_LAYOUT_YPBPR_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// YPbPr //

/// \addtogroup ColorNameModel
/// \{
namespace YPbPr
{
/// \brief Luminance
struct Y_t {};
/// \brief Pb
struct Pb_t {};
/// \brief Pr
struct Pr_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3<
		YPbPr::Y_t,
		YPbPr::Pb_t,
		YPbPr::Pr_t
	> YPbPr_t;


/**
 * @brief YPbPr -> RGB
 */
template < typename SrcP, typename DstP >
void convertYPbPrToRgb( const SrcP& src, DstP& dst )
{
	get_color( dst, red_t()   )	= get_color( src, YPbPr::Y_t() )                                              + 1.402    * get_color( src, YPbPr::Pr_t() );
	get_color( dst, green_t() )	= get_color( src, YPbPr::Y_t() ) - 0.344136 * get_color( src, YPbPr::Pb_t() ) - 0.714136 * get_color( src, YPbPr::Pr_t() );
	get_color( dst, blue_t()  )	= get_color( src, YPbPr::Y_t() ) + 1.772    * get_color( src, YPbPr::Pb_t() )                                             ;
}

/**
 * @brief RGB -> YPbPr
 */
template < typename SrcP, typename DstP >
void convertRgbToYPbPr( const SrcP& src, DstP& dst )
{
	get_color( dst, YPbPr::Y_t()  )	=  0.299    * get_color( src, red_t() ) + 0.587    * get_color( src, green_t() ) + 0.114    * get_color( src, blue_t() );
	get_color( dst, YPbPr::Pb_t() )	= -0.168736 * get_color( src, red_t() ) - 0.331264 * get_color( src, green_t() ) + 0.5      * get_color( src, blue_t() );
	get_color( dst, YPbPr::Pr_t() )	=  0.5      * get_color( src, red_t() ) - 0.418688 * get_color( src, green_t() ) - 0.081312 * get_color( src, blue_t() );
}

}
}
}


#endif
