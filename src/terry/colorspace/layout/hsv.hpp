#ifndef _TERRY_COLOR_LAYOUT_HSV_HPP_
#define	_TERRY_COLOR_LAYOUT_HSV_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// HSV //

/// \addtogroup ColorNameModel
/// \{
namespace hsv
{
/// \brief Hue
struct hue_t {};
/// \brief Saturation
struct saturation_t {};
/// \brief Value
struct value_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< hsl::hue_t
                    , hsv::saturation_t
                    , hsv::value_t
                    > hsv_t;

}
}
}


#endif
