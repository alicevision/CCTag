#ifndef _TERRY_FILTER_CORRELATE_HPP_
#define _TERRY_FILTER_CORRELATE_HPP_

#include "detail/inner_product.hpp"

#include <terry/numeric/operations.hpp>
#include <terry/numeric/assign.hpp>
#include <terry/numeric/init.hpp>
#include <terry/pixel_proxy.hpp>


namespace terry {
namespace filter {

/// @brief 1D un-guarded correlation with a variable-size kernel
template <typename PixelAccum, typename SrcIterator, typename KernelIterator, typename Integer, typename DstIterator>
GIL_FORCEINLINE
DstIterator correlate_pixels_n(
	SrcIterator src_begin,
	SrcIterator src_end,
	KernelIterator ker_begin,
	Integer ker_size,
	DstIterator dst_begin )
{
	using namespace terry::numeric;
	
    typedef typename pixel_proxy<typename std::iterator_traits<SrcIterator>::value_type>::type PIXEL_SRC_REF;
    typedef typename pixel_proxy<typename std::iterator_traits<DstIterator>::value_type>::type PIXEL_DST_REF;
    typedef typename std::iterator_traits<KernelIterator>::value_type kernel_type;
    PixelAccum acc_zero; pixel_zeros_t<PixelAccum>()(acc_zero);

	while( src_begin != src_end )
	{
        pixel_assigns_t<PixelAccum,PIXEL_DST_REF>()(
            std::inner_product(src_begin,src_begin+ker_size,ker_begin,acc_zero,
                               pixel_plus_t<PixelAccum,PixelAccum,PixelAccum>(),
                               pixel_multiplies_scalar_t<PIXEL_SRC_REF,kernel_type,PixelAccum>()),
            *dst_begin);
        ++src_begin; ++dst_begin;
    }
    return dst_begin;
}

/// @brief 1D un-guarded correlation with a fixed-size kernel
template <std::size_t Size,typename PixelAccum,typename SrcIterator,typename KernelIterator,typename DstIterator>
GIL_FORCEINLINE
DstIterator correlate_pixels_k(
	SrcIterator src_begin,
	SrcIterator src_end,
	KernelIterator ker_begin,
	DstIterator dst_begin )
{
	using namespace terry::numeric;
	
    typedef typename pixel_proxy<typename std::iterator_traits<SrcIterator>::value_type>::type PIXEL_SRC_REF;
    typedef typename pixel_proxy<typename std::iterator_traits<DstIterator>::value_type>::type PIXEL_DST_REF;
    typedef typename std::iterator_traits<KernelIterator>::value_type kernel_type;
    PixelAccum acc_zero; pixel_zeros_t<PixelAccum>()(acc_zero);

	while( src_begin != src_end )
	{
        pixel_assigns_t<PixelAccum,PIXEL_DST_REF>()(
            inner_product_k<Size>(src_begin,ker_begin,acc_zero,
                                  pixel_plus_t<PixelAccum,PixelAccum,PixelAccum>(),
                                  pixel_multiplies_scalar_t<PIXEL_SRC_REF,kernel_type,PixelAccum>()),
            *dst_begin);
        ++src_begin; ++dst_begin;
    }
    return dst_begin;
}


}
}


#endif
