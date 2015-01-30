#ifndef _TERRY_NUMERIC_ASSIGN_MINMAX_HPP_
#define _TERRY_NUMERIC_ASSIGN_MINMAX_HPP_

#include "assign.hpp"
#include "operations_assign.hpp"

#include <boost/gil/gil_config.hpp>
#include <boost/gil/pixel.hpp>
#include <boost/gil/color_base_algorithm.hpp>

#include <functional>

namespace terry {
using namespace boost::gil;

namespace numeric {


/// \ingroup PixelNumericOperations
/// \brief construct for setting a pixel to the min channel value (see channel_traits::min_value)
template <typename PixelR> // models pixel concept
struct pixel_assigns_min_t
{
	typedef typename boost::gil::channel_type<PixelR>::type Channel;
	GIL_FORCEINLINE
    PixelR& operator()(PixelR& dst) const
	{
		pixel_assigns_scalar_t<Channel,PixelR>()( channel_traits<Channel>::min_value(), dst);
        return dst;
    }
};

template <typename Pixel>
GIL_FORCEINLINE
void pixel_assigns_min(Pixel& p)
{
    pixel_assigns_min_t<Pixel>()(p);
}

/// \ingroup PixelNumericOperations
/// \brief construct for setting a pixel to the max channel value (see channel_traits::max_value)
template <typename PixelR> // models pixel concept
struct pixel_assigns_max_t
{
	typedef typename boost::gil::channel_type<PixelR>::type Channel;
	GIL_FORCEINLINE
    PixelR& operator() (PixelR& dst) const
	{
		pixel_assigns_scalar_t<Channel,PixelR>()( channel_traits<Channel>::max_value() , dst);
        return dst;
    }
};

template <typename Pixel>
GIL_FORCEINLINE
void pixel_assigns_max(Pixel& p)
{
    pixel_assigns_max_t<Pixel>()(p);
}




}
}

#endif
