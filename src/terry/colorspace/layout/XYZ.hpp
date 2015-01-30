#ifndef _TERRY_COLOR_LAYOUT_XYZ_HPP_
#define	_TERRY_COLOR_LAYOUT_XYZ_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// XYZ //

/// \addtogroup ColorNameModel
/// \{
namespace XYZ
{
/// \brief X
struct X_t {};
/// \brief Y
struct Y_t {};
/// \brief Z
struct Z_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< XYZ::X_t
                    , XYZ::Y_t
                    , XYZ::Z_t
                    > XYZ_t;

}
}
}


#endif
