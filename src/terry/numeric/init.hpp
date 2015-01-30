#ifndef _TERRY_NUMERIC_INIT_HPP_
#define _TERRY_NUMERIC_INIT_HPP_

#include <terry/globals.hpp>

#include <boost/gil/channel.hpp>
#include <boost/gil/pixel.hpp>

#include <functional>

namespace terry {
using namespace boost::gil;

namespace numeric {


/// \ingroup ChannelNumericOperations
/// structure for setting a channel to zero
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel>
struct channel_zeros_t : public std::unary_function<Channel,Channel> {
	GIL_FORCEINLINE
    typename channel_traits<Channel>::reference
    operator()(typename channel_traits<Channel>::reference ch) const {
        return ch=Channel(0);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for setting a pixel to zero (for whatever zero means)
template <typename PixelRef> // models pixel concept
struct pixel_zeros_t {
	GIL_FORCEINLINE
    PixelRef& operator () (PixelRef& p) const {
        static_for_each(p,channel_zeros_t<typename channel_type<PixelRef>::type>());
        return p;
    }
};

template <typename Pixel>
GIL_FORCEINLINE
Pixel& pixel_zeros(Pixel& p)
{
    return pixel_zeros_t<Pixel>()(p);
}

template <typename Pixel>
GIL_FORCEINLINE
Pixel pixel_zeros()
{
	Pixel p;
	return pixel_zeros_t<Pixel>()(p);
}

/// \ingroup ChannelNumericOperations
/// structure for setting a channel to one
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel>
struct channel_ones_t : public std::unary_function<Channel,Channel> {
	GIL_FORCEINLINE
    typename channel_traits<Channel>::reference
    operator()(typename channel_traits<Channel>::reference ch) const {
        return ch=Channel(1);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for setting a pixel to zero (for whatever zero means)
template <typename PixelRef> // models pixel concept
struct pixel_ones_t {
	GIL_FORCEINLINE
    PixelRef& operator () (PixelRef& p) const {
        static_for_each(p,channel_ones_t<typename channel_type<PixelRef>::type>());
        return p;
    }
};

template <typename Pixel>
GIL_FORCEINLINE
Pixel& pixel_ones(Pixel& p)
{
    return pixel_ones_t<Pixel>()(p);
}

template <typename Pixel>
GIL_FORCEINLINE
Pixel pixel_ones()
{
	Pixel p;
	return pixel_ones_t<Pixel>()(p);
}

}
}

#endif
