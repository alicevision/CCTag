#ifndef _TERRY_COLOR_LAYOUT_CMYK_HPP_
#define	_TERRY_COLOR_LAYOUT_CMYK_HPP_

#include "rgb.hpp"

#include <boost/gil/cmyk.hpp>

namespace terry {
namespace color {
namespace layout {

/// \addtogroup ColorNameModel
/// \{
namespace cmyk
{
/// \brief Cyan
using boost::gil::cyan_t;
/// \brief Magenta
using boost::gil::magenta_t;
/// \brief Yellow
using boost::gil::yellow_t;
/// \brief Black
using boost::gil::black_t;
}
/// \}

/// \addtogroup ColorNameModel
/// \{
using boost::gil::cmyk_t;
using boost::gil::cmyk_layout_t;
/// \}



}
}
}


#endif
