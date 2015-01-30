#ifndef _TERRY_COLOR_INVERT_HPP_
#define _TERRY_COLOR_INVERT_HPP_

#include <terry/channel.hpp>

namespace terry {
namespace color {

template< typename Channel>
struct channel_invert_t
{
	void operator()( const Channel& src, Channel& dst ) const
	{
		dst = boost::gil::channel_invert( src );
	}
};

/**
 * @brief Invert colored channel values. So invert all channel values except alpha channel.
 */
struct pixel_invert_colors_t
{
	template<class Pixel>
	Pixel operator()( const Pixel& src ) const
	{
		typedef typename boost::gil::channel_type<Pixel>::type Channel;
		Pixel res;
		boost::gil::static_for_each( src, res, channel_invert_t<Channel>() );
		terry::assign_channel_if_exists_t< Pixel, boost::gil::alpha_t >()( src, res );
		return res;
	}
};

/**
 * @brief Invert channel values
 */
struct pixel_invert_t
{
	template<class Pixel>
	Pixel operator()( const Pixel& src ) const
	{
		typedef typename boost::gil::channel_type<Pixel>::type Channel;
		Pixel res;
		boost::gil::static_for_each( src, res, channel_invert_t<Channel>() );
		return res;
	}
};


}
}

#endif
