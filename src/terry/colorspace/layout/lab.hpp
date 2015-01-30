#ifndef _TERRY_COLOR_LAYOUT_LAB_HPP_
#define	_TERRY_COLOR_LAYOUT_LAB_HPP_

#include "rgb.hpp"

namespace terry {
namespace color {
namespace layout {

////////////////////////////////////////////////////////////////////////////////
// Lab //

/// \addtogroup ColorNameModel
/// \{
namespace lab
{
/// \brief Lightness
struct lightness_t {};
/// \brief chrominance dimension a: green -> magenta
struct a_t {};
/// \brief chrominance dimension a: blue -> yellow
struct b_t {};
}
/// \}

/// \ingroup ColorSpaceModel
typedef boost::mpl::vector3< lab::lightness_t
                    , lab::a_t
                    , lab::b_t
                    > lab_t;

}
}
}


#endif
