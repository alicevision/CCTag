#ifndef _TERRY_COLOR_LAYOUT_HSL_HPP_
#define	_TERRY_COLOR_LAYOUT_HSL_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

/// \addtogroup ColorNameModel
/// \{
namespace hsl
{
/// \brief Hue
struct hue_t {};
/// \brief Saturation
struct saturation_t {};
/// \brief Lightness
struct lightness_t {};
}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< hsl::hue_t
                    , hsl::saturation_t
                    , hsl::lightness_t
                    > hsl_t;
/// \}

}
}
}


#endif
