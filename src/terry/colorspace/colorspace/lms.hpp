#ifndef _TERRY_COLOR_COLORSPACE_LMS_HPP_
#define	_TERRY_COLOR_COLORSPACE_LMS_HPP_

#include "xyz.hpp"

namespace terry {
namespace color {

/// \addtogroup ColorNameModel
/// \{
namespace lms_colorspace
{
/// \brief 
struct L_t {};    
/// \brief 
struct M_t {};
/// \brief 
struct S_t {}; 
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3<
		lms_colorspace::L_t,
		lms_colorspace::M_t,
		lms_colorspace::S_t
	> lms_colorspace_t;

/// \ingroup LayoutModel
typedef layout<lms_colorspace_t> lms_layout_t;

struct LMSParams : public IColorParams
{
	typedef LMSParams This;
	virtual bool operator==( const IColorParams& other ) const
	{
		const This* other_ptr = dynamic_cast<const This*>(&other);
		return other_ptr != NULL;
	}
};
/**
 * @brief LMS colorspace description
 * @todo
 */
struct LMS
{
	typedef XYZ reference;
	typedef LMSParams params;
	
	typedef lms_colorspace_t colorspace;
	typedef lms_layout_t layout;
};

template<typename SChannelType, typename DChannelType>
void color_transform( const LMSParams& params, const pixel<SChannelType,LMS::layout>& src, pixel<DChannelType,XYZ::layout>& dst )
{
	dst = terry::get_black< pixel<DChannelType,XYZ::layout> >();
}
template<typename SChannelType, typename DChannelType>
void color_transform( const LMSParams& params, const pixel<SChannelType,XYZ::layout>& src, pixel<DChannelType,LMS::layout>& dst )
{
	dst = terry::get_black< pixel<DChannelType,LMS::layout> >();
}



}
TERRY_DEFINE_GIL_INTERNALS_3(lms)
TERRY_DEFINE_COLORSPACE_STANDARD_TYPEDEFS(lms)
}

#endif

