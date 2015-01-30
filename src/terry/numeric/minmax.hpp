#ifndef _TERRY_NUMERIC_MINMAX_HPP_
#define _TERRY_NUMERIC_MINMAX_HPP_

#include <boost/gil/gil_config.hpp>
#include <boost/gil/channel.hpp>

#include <functional>

namespace terry {
namespace numeric {

using namespace boost::gil;

/// \ingroup ChannelNumericOperations
/// \brief ch2 = min( ch1, ch2 )
/// structure for adding one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc, typename ChannelDst>
struct channel_assign_min_t : public std::binary_function<ChannelSrc, ChannelDst, ChannelDst>
{
	GIL_FORCEINLINE
	typename channel_traits<ChannelDst>::reference operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
	                                                           typename channel_traits<ChannelDst>::reference ch2 ) const
	{
		return ch2 = std::min( ChannelDst( ch1 ), ch2 );
	}
};


/// \ingroup PixelNumericOperations
/// \brief p2 = min( p1, p2 )
template <typename PixelSrc, // models pixel concept
          typename PixelDst>
// models pixel value concept
struct pixel_assign_min_t
{
	GIL_FORCEINLINE
	PixelDst& operator()( const PixelSrc& p1,
	                      PixelDst& p2 ) const
	{
		static_for_each( p1, p2,
		                 channel_assign_min_t<typename channel_type<PixelSrc>::type,
		                                      typename channel_type<PixelDst>::type>() );
		return p2;
	}
};


/// \ingroup ChannelNumericOperations
/// \brief ch2 = max( ch1, ch2 )
/// this is a generic implementation; user should specialize it for better performance
template <typename ChannelSrc, typename ChannelDst>
struct channel_assign_max_t : public std::binary_function<ChannelSrc, ChannelDst, ChannelDst>
{
	GIL_FORCEINLINE
	typename channel_traits<ChannelDst>::reference operator()( typename channel_traits<ChannelSrc>::const_reference ch1,
	                                                           typename channel_traits<ChannelDst>::reference ch2 ) const
	{
		return ch2 = std::max( ChannelDst( ch1 ), ch2 );
	}
};


/// \ingroup PixelNumericOperations
/// \brief p2 = max( p1, p2 )
template <typename PixelSrc, // models pixel concept
          typename PixelDst>
// models pixel value concept
struct pixel_assign_max_t
{
	GIL_FORCEINLINE
	PixelDst& operator()( const PixelSrc& p1,
	                      PixelDst& p2 ) const
	{
		static_for_each( p1, p2,
		                 channel_assign_max_t<typename channel_type<PixelSrc>::type,
		                                      typename channel_type<PixelDst>::type>() );
		return p2;
	}

};


template<typename CPixel>
struct pixel_minmax_by_channel_t
{
	typedef typename channel_type<CPixel>::type Channel;

	CPixel min;
	CPixel max;

	pixel_minmax_by_channel_t( const CPixel& v )
	: min( v )
	, max( v )
	{
	}

	template<typename Pixel>
	GIL_FORCEINLINE
	void operator()( const Pixel& v )
	{
		pixel_assign_min_t<Pixel,CPixel>()( v, min );
		pixel_assign_max_t<Pixel,CPixel>()( v, max );
	}
};


}
}

#endif
