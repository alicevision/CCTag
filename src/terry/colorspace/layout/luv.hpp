#ifndef _TERRY_COLOR_LAYOUT_LUV_HPP_
#define	_TERRY_COLOR_LAYOUT_LUV_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// Luv //

/// \addtogroup ColorNameModel
/// \{
namespace luv
{
/// \brief Lightness
struct lightness_t {};
/// \brief chrominance dimension u
struct u_t {};
/// \brief chrominance dimension v
struct v_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< luv::lightness_t
                    , luv::u_t
                    , luv::v_t
                    > luv_t;

}
}
}


#endif
