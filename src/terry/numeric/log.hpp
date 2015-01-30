#ifndef _TERRY_NUMERIC_LOG_HPP_
#define _TERRY_NUMERIC_LOG_HPP_

#include <cmath>

namespace terry {
namespace numeric {

template <typename Channel, typename ChannelR>
struct channel_log10_t : public std::unary_function<Channel, ChannelR>
{
	GIL_FORCEINLINE
	ChannelR operator()( typename channel_traits<Channel>::const_reference ch ) const
	{
		return std::log10( ChannelR( ch ) );
	}
};

template <typename PixelRef, typename PixelR = PixelRef> // models pixel concept
struct pixel_log10_t
{
	GIL_FORCEINLINE
	PixelR operator()(const PixelRef & p ) const
	{
		PixelR result;
		boost::gil::static_transform( p, result, channel_log10_t<typename channel_type<PixelRef>::type, typename channel_type<PixelR>::type > ( ) );
		return result;
	}
};


}
}

#endif
