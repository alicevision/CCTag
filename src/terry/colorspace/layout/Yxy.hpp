#ifndef _TERRY_COLOR_LAYOUT_YXY_HPP_
#define	_TERRY_COLOR_LAYOUT_YXY_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// Yxy //

/// \addtogroup ColorNameModel
/// \{
namespace Yxy
{
/// \brief Y
struct Y_t {};
/// \brief x
struct x_t {};
/// \brief y
struct y_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< Yxy::Y_t
                    , Yxy::x_t
                    , Yxy::y_t
                    > Yxy_t;

}
}
}


#endif
