#ifndef _TERRY_NUMERIC_ASSIGN_HPP_
#define _TERRY_NUMERIC_ASSIGN_HPP_

#include <terry/algorithm/for_each.hpp>
#include <terry/pixel_proxy.hpp>

namespace terry {
using namespace boost::gil;

namespace numeric {

/// \ingroup ChannelNumericOperations
/// structure for assigning one channel to another
/// this is a generic implementation; user should specialize it for better performance
template <typename Channel1,typename Channel2>
struct channel_assigns_t : public std::binary_function<Channel1,Channel2,Channel2>
{
	GIL_FORCEINLINE
    typename channel_traits<Channel2>::reference
    operator()( typename channel_traits<Channel1>::const_reference ch1,
                typename channel_traits<Channel2>::reference ch2 ) const
	{
        return ch2=Channel2(ch1);
    }
};

/// \ingroup PixelNumericOperations
///definition and a generic implementation for casting and assigning a pixel to another
///user should specialize it for better performance
template <typename PixelRef,  // models pixel concept
          typename PixelRefR> // models pixel concept
struct pixel_assigns_t
{
	GIL_FORCEINLINE
    PixelRefR& operator()( const PixelRef& src,
                           PixelRefR& dst ) const
	{
        static_for_each(
			src, dst,
			channel_assigns_t<typename channel_type<PixelRef>::type,
							  typename channel_type<PixelRefR>::type>()
			);
        return dst;
    }
};

template <typename Pixel,  // models pixel concept
          typename PixelR> // models pixel concept
GIL_FORCEINLINE
void pixel_assigns(const Pixel& src, Pixel& dst)
{
	pixel_assigns_t<Pixel,PixelR>()( src, dst );
}


/// \ingroup PixelNumericOperations
///definition and a generic implementation for casting and assigning a pixel to another
///user should specialize it for better performance
template <typename Scalar,  // models pixel concept
          typename PixelR> // models pixel concept
struct pixel_assigns_scalar_t
{
	GIL_FORCEINLINE
    PixelR& operator()( const Scalar s,
                        PixelR& dst ) const
	{
        static_for_each(
			dst,
			std::bind1st( channel_assigns_t<Scalar, typename channel_type<PixelR>::type>(), s )
			);
        return dst;
    }
};

template< typename Scalar,  // models pixel concept
          typename PixelR > // models pixel concept
GIL_FORCEINLINE
void pixel_assigns_scalar( const Scalar s, PixelR& dst )
{
	pixel_assigns_scalar_t<Scalar,PixelR>()( s, dst );
}


template< typename SrcIterator, typename DstIterator >
GIL_FORCEINLINE
DstIterator assign_pixels( SrcIterator src, SrcIterator src_end, DstIterator dst )
{
	algorithm::for_each(
		src, src_end, dst,
		pixel_assigns_t< typename pixel_proxy<typename std::iterator_traits<SrcIterator>::value_type>::type,
		                 typename pixel_proxy<typename std::iterator_traits<DstIterator>::value_type>::type >()
		);
	
	return dst + ( src_end - src );
}


}
}

#endif
