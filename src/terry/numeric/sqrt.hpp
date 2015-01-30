#ifndef _TERRY_NUMERIC_SQRT_HPP_
#define _TERRY_NUMERIC_SQRT_HPP_

#include <terry/numeric/operations.hpp>

#include <boost/gil/color_base_algorithm.hpp>

namespace terry {
namespace numeric {

template <typename Channel, typename ChannelR>
struct channel_sqrt_t : public std::unary_function<Channel, ChannelR>
{
	GIL_FORCEINLINE
	ChannelR operator( )( typename channel_traits<Channel>::const_reference ch ) const
	{
		return std::sqrt( ChannelR( ch ) );
	}
};

template <typename PixelRef, typename PixelR = PixelRef> // models pixel concept
struct pixel_sqrt_t
{
	GIL_FORCEINLINE
	PixelR operator ( ) (const PixelRef & p ) const
	{
		PixelR result;
		boost::gil::static_transform( p, result, channel_sqrt_t<typename channel_type<PixelRef>::type, typename channel_type<PixelR>::type > ( ) );
		return result;
	}
};

}
}

#endif
