#ifndef _TERRY_GEOMETRY_SUBIMAGE_HPP_
#define _TERRY_GEOMETRY_SUBIMAGE_HPP_

#include <boost/gil/image_view.hpp>

#include <terry/math/Rect.hpp>

namespace terry {
using namespace boost::gil;

template <typename View, typename T>
inline View subimage_view( const View& src, const Rect<T>& region )
{
	return View( region.x2-region.x1, region.y2-region.y1, src.xy_at(region.x1, region.y1) );
}

}

#endif
