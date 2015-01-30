#ifndef _TERRY_NUMERIC_POW_HPP_
#define _TERRY_NUMERIC_POW_HPP_

#include <terry/globals.hpp>

#include <boost/gil/channel.hpp>
#include <boost/gil/pixel.hpp>
#include <boost/math/special_functions/pow.hpp>

#include <functional>

namespace terry {
using namespace boost::gil;

namespace numeric {

/// \ingroup ChannelNumericOperations
/// structure to compute pow on a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel, int n,typename ChannelR>
struct channel_pow_t : public std::unary_function<Channel,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch) const {
        return boost::math::pow<n>(ChannelR(ch));
    }
};

/// \ingroup PixelNumericOperations
/// \brief construct to compute pow on a pixel
template <typename PixelRef, int n, typename PixelR=PixelRef> // models pixel concept
struct pixel_pow_t {
	GIL_FORCEINLINE
    PixelR operator () (const PixelRef& p) const {
        PixelR result;
        static_transform(p,result, channel_pow_t<typename channel_type<PixelRef>::type,
						                         n,
												 typename channel_type<PixelR>::type>());
        return result;
    }
};

/// \ingroup ChannelNumericOperations
/// structure to compute pow of a scalar by the pixel value
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel,typename Scalar,typename ChannelR>
struct channel_scalar_pow_t : public std::binary_function<Channel,Scalar,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch,
                        const Scalar& s) const {
	typedef typename floating_channel_type_t<ChannelR>::type ChannelRFloat;
        return std::pow(s, ChannelRFloat(ch));
    }
};

/// \ingroup PixelNumericOperations
/// \brief construct to compute pow of a scalar by the pixel value
template <typename PixelRef, // models pixel concept
          typename Scalar,   // models a scalar type
          typename PixelR=PixelRef>   // models pixel value concept
struct pixel_scalar_pow_t {
	GIL_FORCEINLINE
    PixelR operator () (const PixelRef& p,
                        const Scalar& s) const {
        PixelR result;
        static_transform(p,result,
                           std::bind2nd(channel_scalar_pow_t<typename channel_type<PixelRef>::type,
                                                                 Scalar,
                                                                 typename channel_type<PixelR>::type>(),s));
        return result;
    }
};

}
}

#endif
