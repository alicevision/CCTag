#ifndef _TERRY_NUMERIC_SCALAR_HPP_
#define _TERRY_NUMERIC_SCALAR_HPP_

#include "assign.hpp"

namespace terry {
namespace numeric {


/// @brief destination is set to be product of the source and a scalar
template <typename PixelAccum, typename SrcView, typename Scalar, typename DstView>
GIL_FORCEINLINE
void view_multiplies_scalar( const SrcView& src, const Scalar& scalar, const DstView& dst )
{
    assert( src.dimensions() == dst.dimensions() );

	typedef typename pixel_proxy<typename SrcView::value_type>::type PIXEL_SRC_REF;
    typedef typename pixel_proxy<typename DstView::value_type>::type PIXEL_DST_REF;
	
    int height = src.height();
    for( int rr = 0; rr < height; ++rr )
	{
        typename SrcView::x_iterator it_src = src.row_begin(rr);
        typename DstView::x_iterator it_dst = dst.row_begin(rr);
        typename SrcView::x_iterator it_src_end = src.row_end(rr);
        while( it_src != it_src_end )
		{
            pixel_assigns_t<PixelAccum,PIXEL_DST_REF>()(
                pixel_multiplies_scalar_t<PIXEL_SRC_REF,Scalar,PixelAccum>()( *it_src, scalar ),
                *it_dst );
			
            ++it_src; ++it_dst;
        }
    }
}

}
}

#endif
