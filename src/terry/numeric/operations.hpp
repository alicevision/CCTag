#ifndef _TERRY_NUMERIC_OPERATIONS_HPP_
#define _TERRY_NUMERIC_OPERATIONS_HPP_

#include <terry/globals.hpp>

#include <boost/gil/channel.hpp>
#include <boost/gil/pixel.hpp>

#include <functional>

namespace terry {
using namespace boost::gil;

namespace numeric {

/// \ingroup ChannelNumericOperations
/// structure for adding one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel1,typename Channel2,typename ChannelR>
struct channel_plus_t : public std::binary_function<Channel1,Channel2,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel1>::const_reference ch1,
                        typename channel_traits<Channel2>::const_reference ch2) const {
        return ChannelR(ch1)+ChannelR(ch2);
    }
};

/// \ingroup PixelNumericOperations
/// \brief construct for adding two pixels
template <typename PixelRef1, // models pixel concept
          typename PixelRef2, // models pixel concept
          typename PixelR>    // models pixel value concept
struct pixel_plus_t {
	GIL_FORCEINLINE
    PixelR operator() (const PixelRef1& p1,
                       const PixelRef2& p2) const {
        PixelR result;
        static_transform(p1,p2,result,
                           channel_plus_t<typename channel_type<PixelRef1>::type,
                                          typename channel_type<PixelRef2>::type,
                                          typename channel_type<PixelR>::type>());
        return result;
    }
};

/// \ingroup ChannelNumericOperations
/// structure for subtracting one channel from another
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel1,typename Channel2,typename ChannelR>
struct channel_minus_t : public std::binary_function<Channel1,Channel2,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel1>::const_reference ch1,
                        typename channel_traits<Channel2>::const_reference ch2) const {
        return ChannelR(ch1)-ChannelR(ch2);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for subtracting two pixels
template <typename PixelRef1, // models pixel concept
          typename PixelRef2, // models pixel concept
          typename PixelR>    // models pixel value concept
struct pixel_minus_t {
	GIL_FORCEINLINE
    PixelR operator() (const PixelRef1& p1,
                       const PixelRef2& p2) const {
        PixelR result;
        static_transform(p1,p2,result,
                           channel_minus_t<typename channel_type<PixelRef1>::type,
                                           typename channel_type<PixelRef2>::type,
                                           typename channel_type<PixelR>::type>());
        return result;
    }
};


/// \ingroup ChannelNumericOperations
/// structure for multiplying one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel1,typename Channel2,typename ChannelR>
struct channel_multiplies_t : public std::binary_function<Channel1,Channel2,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel1>::const_reference ch1,
                        typename channel_traits<Channel2>::const_reference ch2) const {
        return ChannelR(ch1)*ChannelR(ch2);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for adding two pixels
template <typename PixelRef1, // models pixel concept
          typename PixelRef2, // models pixel concept
          typename PixelR>    // models pixel value concept
struct pixel_multiplies_t {
	GIL_FORCEINLINE
    PixelR operator() (const PixelRef1& p1,
                       const PixelRef2& p2) const {
        PixelR result;
        static_transform(p1,p2,result,
                           channel_multiplies_t<typename channel_type<PixelRef1>::type,
                                          typename channel_type<PixelRef2>::type,
                                          typename channel_type<PixelR>::type>());
        return result;
    }
};


/// \ingroup ChannelNumericOperations
/// structure for dividing channels
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel1,typename Channel2,typename ChannelR>
struct channel_divides_t : public std::binary_function<Channel1,Channel2,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel1>::const_reference ch1,
                        typename channel_traits<Channel2>::const_reference ch2) const {
        return ChannelR(ch1)/ChannelR(ch2);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for subtracting two pixels
template <typename PixelRef1, // models pixel concept
          typename PixelRef2, // models pixel concept
          typename PixelR>    // models pixel value concept
struct pixel_divides_t {
	GIL_FORCEINLINE
    PixelR operator() (const PixelRef1& p1,
                       const PixelRef2& p2) const {
        PixelR result;
        static_transform(p1,p2,result,
                           channel_divides_t<typename channel_type<PixelRef1>::type,
                                           typename channel_type<PixelRef2>::type,
                                           typename channel_type<PixelR>::type>());
        return result;
    }
};

/// \ingroup ChannelNumericOperations
/// structure for adding a scalar to a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel,typename Scalar,typename ChannelR>
struct channel_plus_scalar_t : public std::binary_function<Channel,Scalar,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch,
                        const Scalar& s) const {
        return ChannelR(ch)+ChannelR(s);
    }
};

/// \ingroup ChannelNumericOperations
/// structure for subtracting a scalar from a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel,typename Scalar,typename ChannelR>
struct channel_minus_scalar_t : public std::binary_function<Channel,Scalar,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch,
                        const Scalar& s) const {
        return ChannelR(ch-s);
    }
};

/// \ingroup ChannelNumericOperations
/// structure for multiplying a scalar to one channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel,typename Scalar,typename ChannelR>
struct channel_multiplies_scalar_t : public std::binary_function<Channel,Scalar,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch,
                        const Scalar& s) const {
        return ChannelR(ch)*ChannelR(s);
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for multiplying scalar to a pixel
template <typename PixelRef, // models pixel concept
          typename Scalar,   // models a scalar type
          typename PixelR=PixelRef>   // models pixel value concept
struct pixel_multiplies_scalar_t {
	GIL_FORCEINLINE
    PixelR operator () (const PixelRef& p,
                        const Scalar& s) const {
        PixelR result;
        static_transform(p,result,
                           std::bind2nd(channel_multiplies_scalar_t<typename channel_type<PixelRef>::type,
                                                                    Scalar,
                                                                    typename channel_type<PixelR>::type>(),s));
        return result;
    }
};


/// \ingroup ChannelNumericOperations
/// structure for dividing a channel by a scalar
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel,typename Scalar,typename ChannelR>
struct channel_divides_scalar_t : public std::binary_function<Channel,Scalar,ChannelR> {
	GIL_FORCEINLINE
    ChannelR operator()(typename channel_traits<Channel>::const_reference ch,
                        const Scalar& s) const {
        return ChannelR(ch)/ChannelR(s);
    }
};

/// \ingroup PixelNumericOperations
/// \brief construct for dividing a pixel by a scalar
template <typename PixelRef, // models pixel concept
          typename Scalar,   // models a scalar type
          typename PixelR=PixelRef>   // models pixel value concept
struct pixel_divides_scalar_t {
	GIL_FORCEINLINE
    PixelR operator () (const PixelRef& p,
                        const Scalar& s) const {
        PixelR result;
        static_transform(p,result,
                           std::bind2nd(channel_divides_scalar_t<typename channel_type<PixelRef>::type,
                                                                 Scalar,
                                                                 typename channel_type<PixelR>::type>(),s));
        return result;
    }
};


/// \ingroup ChannelNumericOperations
/// structure for halving a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel>
struct channel_halves_t : public std::unary_function<Channel,Channel> {
	GIL_FORCEINLINE
    typename channel_traits<Channel>::reference
    operator()(typename channel_traits<Channel>::reference ch) const {
        return ch/=2.0;
    }
};


/// \ingroup PixelNumericOperations
/// \brief construct for dividing a pixel by 2
template <typename PixelRef> // models pixel concept
struct pixel_halves_t {
	GIL_FORCEINLINE
    PixelRef& operator () (PixelRef& p) const {
        static_for_each(p,channel_halves_t<typename channel_type<PixelRef>::type>());
        return p;
    }
};

template <typename Pixel>
GIL_FORCEINLINE
void pixel_halves(Pixel& p)
{
    pixel_halves_t<Pixel>()(p);
}


}
}

#endif
