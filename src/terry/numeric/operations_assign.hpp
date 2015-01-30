#ifndef _TERRY_NUMERIC_OPERATIONS_ASSIGN_HPP_
#define _TERRY_NUMERIC_OPERATIONS_ASSIGN_HPP_

#include <boost/gil/gil_config.hpp>
#include <boost/gil/channel.hpp>

#include <functional>

namespace terry {
using namespace boost::gil;

namespace numeric {


/// \ingroup ChannelNumericOperations
/// \brief ch2 += ch1
/// structure for adding one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc,typename ChannelDst>
struct channel_plus_assign_t : public std::binary_function<ChannelSrc,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
                typename channel_traits<ChannelDst>::reference ch2 ) const {
        return ch2 += ChannelDst( ch1 );
    }
};

/// \ingroup PixelNumericOperations
/// \brief p2 += p1
template <typename PixelSrc, // models pixel concept
          typename PixelDst = PixelSrc> // models pixel concept
struct pixel_plus_assign_t {
	GIL_FORCEINLINE
    PixelDst& operator()( const PixelSrc& p1,
                          PixelDst& p2 ) const {
        static_for_each( p1, p2,
                         channel_plus_assign_t<typename channel_type<PixelSrc>::type,
                                               typename channel_type<PixelDst>::type>() );
        return p2;
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch2 -= ch1
/// structure for subtracting one channel from another
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc,typename ChannelDst>
struct channel_minus_assign_t : public std::binary_function<ChannelSrc,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
                typename channel_traits<ChannelDst>::reference ch2 ) const {
        return ch2 -= ChannelDst( ch1 );
    }
};

/// \ingroup PixelNumericOperations
/// \brief p2 -= p1
template <typename PixelSrc, // models pixel concept
          typename PixelDst = PixelSrc> // models pixel concept
struct pixel_minus_assign_t {
	GIL_FORCEINLINE
    PixelDst& operator()( const PixelSrc& p1,
                          PixelDst& p2 ) const {
        static_for_each( p1, p2,
                         channel_minus_assign_t<typename channel_type<PixelSrc>::type,
                                                typename channel_type<PixelDst>::type>() );
        return p2;
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch2 *= ch1
/// structure for multiplying one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc,typename ChannelDst>
struct channel_multiplies_assign_t : public std::binary_function<ChannelSrc,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
                typename channel_traits<ChannelDst>::reference ch2 ) const {
        return ch2 *= ch1;
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch2 /= ch1
/// structure for dividing channels
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc,typename ChannelDst>
struct channel_divides_assign_t : public std::binary_function<ChannelSrc,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
                typename channel_traits<ChannelDst>::reference ch2 ) const {
        return ch2 /= ch1;
    }
};


/// \ingroup ChannelNumericOperations
/// \brief ch += s
/// structure for adding a scalar to a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Scalar, typename ChannelDst>
struct channel_plus_scalar_assign_t : public std::binary_function<Scalar,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( const Scalar& s,
	            typename channel_traits<ChannelDst>::reference ch ) const {
        return ch += ChannelDst(s);
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch -= s
/// structure for subtracting a scalar from a channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Scalar, typename ChannelDst>
struct channel_minus_scalar_assign_t : public std::binary_function<Scalar,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( const Scalar& s,
	            typename channel_traits<ChannelDst>::reference ch ) const {
        return ch -= ChannelDst(s);
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch *= s
/// structure for multiplying a scalar to one channel
/// this is a generic implementation; user should specialize it for better performance
template <typename Scalar, typename ChannelDst>
struct channel_multiplies_scalar_assign_t : public std::binary_function<Scalar,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( const Scalar& s,
	            typename channel_traits<ChannelDst>::reference ch ) const {
        return ch *= s;
    }
};

/// \ingroup PixelNumericOperations
/// \brief p *= s
template <typename Scalar, // models a scalar type
	      typename PixelDst>  // models pixel concept
struct pixel_multiplies_scalar_assign_t {
	GIL_FORCEINLINE
    PixelDst& operator()( const Scalar& s,
	                      PixelDst& p ) const {
        static_for_each( p, std::bind1st( channel_multiplies_scalar_assign_t<Scalar, typename channel_type<PixelDst>::type>(), s ) );
		return p;
    }
};

/// \ingroup ChannelNumericOperations
/// \brief ch /= s
/// structure for dividing a channel by a scalar
/// this is a generic implementation; user should specialize it for better performance
template <typename Scalar, typename ChannelDst>
struct channel_divides_scalar_assign_t : public std::binary_function<Scalar,ChannelDst,ChannelDst> {
	GIL_FORCEINLINE
    typename channel_traits<ChannelDst>::reference
	operator()( const Scalar& s,
	            typename channel_traits<ChannelDst>::reference ch ) const {
        return ch /= s;
    }
};

/// \ingroup PixelNumericOperations
/// \brief p /= s
template <typename Scalar, // models a scalar type
	      typename PixelDst>  // models pixel concept
struct pixel_divides_scalar_assign_t
{
	GIL_FORCEINLINE
    PixelDst& operator()( const Scalar& s,
	                      PixelDst& p ) const
	{
        static_for_each( p, std::bind1st( channel_divides_scalar_assign_t<Scalar, typename channel_type<PixelDst>::type>(), s ) );
		return p;
    }
};

}
}

#endif
