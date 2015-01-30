#ifndef _TERRY_COPY_HPP_
#define	_TERRY_COPY_HPP_

#include <boost/gil/color_base_algorithm.hpp>
#include <terry/channel_view.hpp>

namespace terry {

namespace detail_copy_channel {

template<class Channel, class View>
void copy_channel_if_exist( const View& src, const View& dst, const boost::mpl::true_ )
{
	using namespace boost::gil;
	copy_pixels( channel_view<Channel>(src), channel_view<Channel>(dst) );
}
template<class Channel, class View>
void copy_channel_if_exist( const View& src, const View& dst, const boost::mpl::false_ )
{
}

template<class Channel, class Pixel>
void copy_pixel_channel_if_exist( const Pixel& src, const Pixel& dst, const boost::mpl::true_ )
{
	using namespace boost::gil;
	get_color( dst, Channel() )	=  get_color( src, Channel() );
}
template<class Channel, class Pixel>
void copy_pixel_channel_if_exist( const Pixel& src, const Pixel& dst, const boost::mpl::false_ )
{
}

}

template<class Channel, class View>
void copy_channel_if_exist( const View& src, const View& dst )
{
	using namespace boost::gil;
	typedef typename contains_color<typename View::value_type, Channel>::type hasChannel;
	detail_copy_channel::copy_channel_if_exist<Channel, View>( src, dst, hasChannel() );
}

template<class Channel, class Pixel>
void copy_pixel_channel_if_exist( const Pixel& src, const Pixel& dst )
{
	using namespace boost::gil;
	typedef typename contains_color<Pixel, Channel>::type hasChannel;
	detail_copy_channel::copy_pixel_channel_if_exist<Channel, Pixel>( src, dst, hasChannel() );
}



}

#endif

