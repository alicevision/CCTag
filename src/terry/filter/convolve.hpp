#ifndef _TERRY_FILTER_CONVOLVE_HPP_
#define _TERRY_FILTER_CONVOLVE_HPP_

#include "correlate.hpp"
#include "detail/kernel.hpp"

#include <terry/numeric/scalar.hpp>
#include <terry/numeric/init.hpp>
#include <terry/numeric/assign.hpp>

#include <boost/gil/gil_config.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/algorithm.hpp>
#include <boost/gil/metafunctions.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/mpl/bool.hpp>

#include <cstddef>
#include <cassert>
#include <algorithm>
#include <vector>
#include <functional>


namespace terry {
using namespace boost::gil;

namespace filter {

/// @ingroup ImageAlgorithms
/// Boundary options for 1D correlations/convolutions
enum convolve_boundary_option  {
    convolve_option_output_ignore,   /// do nothing to the output
    convolve_option_output_zero,     /// set the output to zero
    convolve_option_extend_padded,   /// assume the source boundaries to be padded already
    convolve_option_extend_zero,     /// assume the source boundaries to be zero
    convolve_option_extend_constant, /// assume the source boundaries to be the boundary value
    convolve_option_extend_mirror    /// assume the source boundaries to be the mirror of source
};

namespace detail {

/// @ingroup PointModel
template <typename T> GIL_FORCEINLINE
bool operator>=(const point2<T>& p1, const point2<T>& p2) { return (p1.x>=p2.x && p1.y>=p2.y); }
/// @ingroup PointModel
template <typename T> GIL_FORCEINLINE
bool operator<=(const point2<T>& p1, const point2<T>& p2) { return (p1.x<=p2.x && p1.y<=p2.y); }
/// @ingroup PointModel
template <typename T, typename T2> GIL_FORCEINLINE
bool operator>=(const point2<T>& p, const T2 v) { return (p.x>=v && p.y>=v); }
/// @ingroup PointModel
template <typename T, typename T2> GIL_FORCEINLINE
bool operator<=(const point2<T>& p, const T2 v) { return (p.x<=v && p.y<=v); }

/// compute the correlation of 1D kernel with the rows of an image
/// @param src source view
/// @param dst destination view
/// @param ker dynamic size kernel
/// @param dst_tl topleft point of dst in src coordinates. Must be Point(0,0) if src.dimensions()==dst.dimensions().
///        We can see it as a vector to move dst in src coordinates.
/// @param option boundary option
/// @param correlator correlator functor
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView,typename Correlator>
void correlate_rows_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                         const convolve_boundary_option option,
                         Correlator correlator )
{
	using namespace terry::numeric;
	
	// assert dst frame with shift is inside src frame
    assert( src.dimensions() >= dst.dimensions() );
	// dst must be contained in src
    assert( dst_tl >= 0 );
    assert( dst_tl <= src.dimensions() );
	assert( dst_tl + dst.dimensions() <= src.dimensions() );
    assert( ker.size() != 0 );

	typedef typename SrcView::point_t point_t;
	typedef typename point_t::value_type coord_t;
    typedef typename pixel_proxy<typename SrcView::value_type>::type PIXEL_SRC_REF;
    typedef typename pixel_proxy<typename DstView::value_type>::type PIXEL_DST_REF;
    typedef typename Kernel::value_type kernel_type;

    if( ker.size() == 1 )
	{
		// reduces to a multiplication
		view_multiplies_scalar<PixelAccum>(
				subimage_view(src, dst_tl, dst.dimensions()),
				*ker.begin(),
				dst
			);
        return;
    }

    if( dst.dimensions().x == 0 || dst.dimensions().y == 0 )
		return;
	
	//  ................................................................
	//  .                     src with kernel size adds                .
	//  .                                                              .
	//  .          _________________________________________           .
	//  .          |             src and dst               |           .
	//  .          |          ____________                 |           .
	// <. left_out | left_in |            | right_in       | right_out .>
	//  .          |         |    roi     |                |           .
	//  .          |         |            |                |           .
	//  .          |         |____________|                |           .
	//  .          |_______________________________________|           .
	//  .                                                              .
	//  ................................................................
	// < > : represents the temporary buffer
	const point_t dst_br  = dst_tl + dst.dimensions();
	const coord_t left_in   = std::min(boost::numeric_cast<coord_t>(ker.left_size()), dst_tl.x);
	const coord_t left_out  = std::max(boost::numeric_cast<coord_t>(ker.left_size()) - dst_tl.x, (coord_t)0);
	const coord_t right_tmp = src.dimensions().x - dst_br.x;
	const coord_t right_in  = std::min(boost::numeric_cast<coord_t>(ker.right_size()), right_tmp);
	const coord_t right_out = std::max(boost::numeric_cast<coord_t>(ker.right_size()) - right_tmp, (coord_t)0);

	const coord_t srcRoi_left = dst_tl.x - left_in;
	const coord_t srcRoi_right = dst_br.x + right_in;
	const coord_t srcRoi_width = dst.dimensions().x + left_in + right_in;

    PixelAccum acc_zero; pixel_zeros_t<PixelAccum>()(acc_zero);

	if( option == convolve_option_output_ignore || option == convolve_option_output_zero )
	{
        typename DstView::value_type dst_zero; pixel_assigns_t<PixelAccum,PIXEL_DST_REF>()(acc_zero,dst_zero);
        if( dst.dimensions().x < static_cast<coord_t>(ker.size()) )
		{
            if( option == convolve_option_output_zero )
                fill_pixels( dst, dst_zero );
        }
		else
		{
			std::vector<PixelAccum> buffer(srcRoi_width);
            for( coord_t yy = 0; yy < dst.dimensions().y; ++yy )
			{
				coord_t yy_src = yy + dst_tl.y;
                assign_pixels( src.x_at(srcRoi_left,yy_src),
				               src.x_at(srcRoi_right,yy_src),
							   &buffer.front() );

                typename DstView::x_iterator it_dst=dst.row_begin(yy);
                if (option==convolve_option_output_zero)
                    std::fill_n(it_dst,left_out,dst_zero);
				it_dst += left_out;

				const int buffer_dst_size = dst.dimensions().x - left_out-right_out;
				correlator( &buffer.front(), &buffer.front() + buffer_dst_size, // why not always use begin(), does front() have a performance impact ?
                            ker.begin(), it_dst );
                it_dst += buffer_dst_size;

                if( option == convolve_option_output_zero )
                    std::fill_n( it_dst, right_out, dst_zero );
            }
        }
    }
	else
	{
        std::vector<PixelAccum> buffer( dst.dimensions().x + (ker.size() - 1) );
        for( int yy=0; yy<dst.dimensions().y; ++yy )
		{
			coord_t yy_src = yy + dst_tl.y;
			// fill buffer from src view depending on boundary option
            switch( option )
			{
				case convolve_option_extend_padded:
				{
					assign_pixels( src.x_at(dst_tl.x-ker.left_size(),yy_src),
								   src.x_at(dst_br.x+ker.right_size(),yy_src),
								   &buffer.front() );
					break;
				}
				case convolve_option_extend_zero:
				{
					PixelAccum* it_buffer=&buffer.front();
					std::fill_n(it_buffer,left_out,acc_zero);
					it_buffer += left_out;
					
					it_buffer = assign_pixels(src.x_at(srcRoi_left,yy_src),src.x_at(srcRoi_right,yy_src),it_buffer);

					std::fill_n(it_buffer,right_out,acc_zero);
					break;
				}
				case convolve_option_extend_constant:
				{
					PixelAccum* it_buffer=&buffer.front();
					PixelAccum filler;
					pixel_assigns_t<PIXEL_SRC_REF,PixelAccum>()(*src.x_at(srcRoi_left,yy_src),filler);
					std::fill_n(it_buffer,left_out,filler);
					it_buffer += left_out;

					it_buffer = assign_pixels(src.x_at(srcRoi_left,yy_src),src.x_at(srcRoi_right,yy_src),it_buffer);

					pixel_assigns_t<PIXEL_SRC_REF,PixelAccum>()(*src.x_at(srcRoi_right-1,yy_src),filler);
					std::fill_n(it_buffer,right_out,filler);
					break;
				}
				case convolve_option_extend_mirror:
				{
					PixelAccum* it_buffer = &buffer.front();
					typedef typename SrcView::reverse_iterator reverse_iterator;
					const unsigned int nleft = boost::numeric_cast<unsigned int>(left_out / srcRoi_width);
					coord_t copy_size = buffer.size();
					const coord_t left_rest = left_out % srcRoi_width;
					bool reverse;
					if( nleft % 2 ) // odd
					{
						assign_pixels( src.at(srcRoi_right-1-left_rest,yy_src),
									   src.at(srcRoi_right-1,yy_src),
									   it_buffer );
						reverse = true; // next step reversed
					}
					else
					{
						assign_pixels( reverse_iterator(src.at(srcRoi_left+left_rest,yy_src)),
									   reverse_iterator(src.at(srcRoi_left,yy_src)),
									   it_buffer );
						reverse = false; // next step not reversed
					}
					it_buffer += left_rest;
					copy_size -= left_rest;
					while( copy_size )
					{
						coord_t tmp_size;
						if( copy_size > srcRoi_width ) // if kernel left size > src width... (extrem case)
							tmp_size = srcRoi_width;
						else // standard case
							tmp_size = copy_size;
						
						if( reverse )
						{
							assign_pixels( reverse_iterator(src.at(srcRoi_right,yy_src)),
										   reverse_iterator(src.at(srcRoi_right-tmp_size,yy_src)),
										   it_buffer );
						}
						else
						{
							assign_pixels( src.at(srcRoi_left,yy_src),
										   src.at(srcRoi_left+tmp_size,yy_src),
										   it_buffer );
						}
						it_buffer += tmp_size;
						copy_size -= tmp_size;
						reverse = !reverse;
					}
					break;
				}
				case convolve_option_output_ignore:
				case convolve_option_output_zero:
					assert(false);
            }
            correlator( &buffer.front(),&buffer.front()+dst.dimensions().x,
                        ker.begin(),
                        dst.row_begin(yy) );
        }
    }
}

template <typename PixelAccum>
class correlator_n
{
private:
    std::size_t _size;
public:
    correlator_n(std::size_t size_in) : _size(size_in) {}
    template <typename SrcIterator,typename KernelIterator,typename DstIterator>
	GIL_FORCEINLINE
    void operator()(SrcIterator src_begin,SrcIterator src_end,
                    KernelIterator ker_begin,
                    DstIterator dst_begin) {
        correlate_pixels_n<PixelAccum>(src_begin,src_end,ker_begin,_size,dst_begin);
    }
};

template <std::size_t Size,typename PixelAccum>
struct correlator_k
{
public:
    template <typename SrcIterator,typename KernelIterator,typename DstIterator>
	GIL_FORCEINLINE
    void operator()(SrcIterator src_begin,SrcIterator src_end,
                    KernelIterator ker_begin,
                    DstIterator dst_begin){
        correlate_pixels_k<Size,PixelAccum>(src_begin,src_end,ker_begin,dst_begin);
    }
};

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                   const convolve_boundary_option option, const boost::mpl::true_ rows, const boost::mpl::false_ /*fixed*/ )
{
	correlate_rows_imp<PixelAccum>(src,ker,dst,dst_tl,option,detail::correlator_n<PixelAccum>(ker.size()));
}

/// @ingroup ImageAlgorithms
/// correlate a 1D fixed-size kernel along the rows of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                   const convolve_boundary_option option, const boost::mpl::true_ rows, const boost::mpl::true_ /*fixed*/ )
{
	correlate_rows_imp<PixelAccum>(src,ker,dst,dst_tl,option,detail::correlator_k<Kernel::static_size,PixelAccum>());
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the columns of an image
/// can be remove with "fixed" param as template argument
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                   const convolve_boundary_option option, const boost::mpl::false_ rows, const boost::mpl::false_ fixed )
{
	correlate_1d_imp<PixelAccum>( transposed_view(src), ker, transposed_view(dst), typename SrcView::point_t(dst_tl.y, dst_tl.x), option, boost::mpl::true_(), fixed );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D fixed-size kernel along the columns of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                   const convolve_boundary_option option, const boost::mpl::false_ rows, const boost::mpl::true_ fixed )
{
	correlate_1d_imp<PixelAccum>( transposed_view(src), ker, transposed_view(dst), typename SrcView::point_t(dst_tl.y, dst_tl.x), option, boost::mpl::true_(), fixed );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D fixed-size kernel along the rows of an image
template <bool rows, typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_auto_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                          const convolve_boundary_option option, const boost::mpl::true_ fixed )
{
	// already a fixed-size kernel, so nothing special to do for the auto conversion variable-size to fixed-size kernel
	correlate_1d_imp<PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option, boost::mpl::bool_<rows>(), fixed );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_auto_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                          const convolve_boundary_option option, const boost::mpl::false_ /*fixed*/ )
{
	typedef boost::mpl::bool_<rows> Rows;
	typedef boost::mpl::true_ Fixed;

	switch( ker.size() )
	{
		case 3:
		{
			typedef kernel_1d_fixed<typename Kernel::value_type, 3> FixedKernel;
			FixedKernel fker( ker.begin(), ker.center() );
			correlate_1d_imp<PixelAccum,SrcView,FixedKernel,DstView>( src, fker, dst, dst_tl, option, Rows(), Fixed() );
			break;
		}
		case 5:
		{
			typedef kernel_1d_fixed<typename Kernel::value_type, 5> FixedKernel;
			FixedKernel fker( ker.begin(), ker.center() );
			correlate_1d_imp<PixelAccum,SrcView,FixedKernel,DstView>( src, fker, dst, dst_tl, option, Rows(), Fixed() );
			break;
		}
		case 7:
		{
			typedef kernel_1d_fixed<typename Kernel::value_type, 7> FixedKernel;
			FixedKernel fker( ker.begin(), ker.center() );
			correlate_1d_imp<PixelAccum,SrcView,FixedKernel,DstView>( src, fker, dst, dst_tl, option, Rows(), Fixed() );
			break;
		}
		case 9:
		{
			typedef kernel_1d_fixed<typename Kernel::value_type, 9> FixedKernel;
			FixedKernel fker( ker.begin(), ker.center() );
			correlate_1d_imp<PixelAccum,SrcView,FixedKernel,DstView>( src, fker, dst, dst_tl, option, Rows(), Fixed() );
			break;
		}
		case 11:
		{
			typedef kernel_1d_fixed<typename Kernel::value_type, 11> FixedKernel;
			FixedKernel fker( ker.begin(), ker.center() );
			correlate_1d_imp<PixelAccum,SrcView,FixedKernel,DstView>( src, fker, dst, dst_tl, option, Rows(), Fixed() );
			break;
		}
		default:
		{
			correlate_1d_imp<PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option, Rows(), boost::mpl::false_() );
			break;
		}
	}
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_auto( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                        const convolve_boundary_option option, const boost::mpl::true_ autoEnabled )
{
	typedef typename Kernel::is_fixed_size_t Fixed;
	correlate_1d_auto_imp<rows,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option, Fixed() );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_auto( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                        const convolve_boundary_option option, const boost::mpl::false_ autoEnabled )
{
	typedef typename Kernel::is_fixed_size_t Fixed;
	typedef boost::mpl::bool_<rows> Rows;
	correlate_1d_imp<PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option, Rows(), Fixed() );
}

} // namespace detail //


/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool autoEnabled, bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_imp( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                        const convolve_boundary_option option = convolve_option_extend_zero )
{
	detail::correlate_1d_auto<rows,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option, boost::mpl::bool_<autoEnabled>() );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d_auto( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                        const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_1d_imp<true,rows,PixelAccum,rows,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool rows, typename PixelAccum, typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_1d( const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                        const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_1d_imp<false,rows,PixelAccum,rows,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option);
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <bool autoEnabled,typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_rows_imp(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_1d_imp<autoEnabled,true,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_rows_auto(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_rows_imp<true,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the rows of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_rows(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl = typename SrcView::point_t(0,0),
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_rows_imp<false,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the columns of an image
template <bool autoEnabled,typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_cols_imp(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_1d_imp<autoEnabled,false,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the columns of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_cols_auto(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl,
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_cols_imp<true,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 1D variable-size kernel along the columns of an image
template <typename PixelAccum,typename SrcView,typename Kernel,typename DstView>
GIL_FORCEINLINE
void correlate_cols(const SrcView& src, const Kernel& ker, const DstView& dst, const typename SrcView::point_t& dst_tl = typename SrcView::point_t(0,0),
                    const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_cols_imp<false,PixelAccum,SrcView,Kernel,DstView>( src, ker, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 2D separable variable-size kernel (kernelX and kernelY)
template <bool autoEnabled, typename PixelAccum, template<typename> class Alloc, typename SrcView, typename KernelX,typename KernelY,typename DstView >
GIL_FORCEINLINE
void correlate_rows_cols_imp( const SrcView& src,
                          const KernelX& kernelX,
						  const KernelY& kernelY,
						  const DstView& dst,
						  const typename SrcView::point_t& dst_tl,
						  const convolve_boundary_option option = convolve_option_extend_zero )
{
	// dst with dst_tl shift must be inside src
	BOOST_ASSERT( dst.width() + dst_tl.x <= src.width() );
	BOOST_ASSERT( dst.height() + dst_tl.y <= src.height() );
    typedef typename DstView::point_t Point;
    typedef typename DstView::coord_t Coord;
	typedef typename view_type_from_pixel<PixelAccum, is_planar<DstView>::value >::type ViewAccum;
	typedef image<PixelAccum, is_planar<DstView>::value, Alloc<unsigned char> > ImageAccum;

	if( kernelX.size() > 2 && kernelY.size() > 2 )
	{
		if( dst.dimensions() == src.dimensions() ) // no tiles... easy !
		{
			typename SrcView::point_t zero(0,0);
			correlate_rows_imp<autoEnabled,PixelAccum>( src, kernelX, dst, zero, option );
			correlate_cols_imp<autoEnabled,PixelAccum>( dst, kernelY, dst, zero, option );
		}
		else
		{
			// we have 2 pass, so to use tiles, we need a temporary buffer
			// _________________src______________
			// |      ....proc_src_roi.....      |
			// |      :  :             :  :      |
			// |      :  :_____________:  :      |
			// |      :  |             |  :      |
			// |      :  |     dst     |  :      |
			// |      :  |             |  :      |
			// |      :  |_____________|  :      |
			// |      :  : tmp_buffer  :  :      |
			// |      :..:.............:..:      |
			// |_________________________________|
			// tmp_buffer: is the temporary buffer used after the correlate_rows
			//             (width of procWin and height of proc_src_roi)
			Coord top_in = std::min( boost::numeric_cast<Coord>( kernelY.left_size()), dst_tl.y );
			Coord bottom_in = std::min( boost::numeric_cast<Coord>( kernelY.right_size()), src.height()-(dst_tl.y+dst.height()) );
			Point image_tmp_size( dst.dimensions() );
			image_tmp_size.y += top_in + bottom_in;
			Point image_tmp_tl( dst_tl );
			image_tmp_tl.y -= top_in;
			
			ImageAccum image_tmp( image_tmp_size );
			ViewAccum view_tmp = view( image_tmp );
			const Point dst_tmp_tl( 0, top_in );

			correlate_rows_imp<autoEnabled,PixelAccum>( src, kernelX, view_tmp, image_tmp_tl, option );
			correlate_cols_imp<autoEnabled,PixelAccum>( view_tmp, kernelY, dst, dst_tmp_tl, option );
		}
	}
	else if( kernelX.size() > 2 )
	{
		correlate_rows_imp<autoEnabled,PixelAccum>( src, kernelX, dst, dst_tl, option );
	}
	else if( kernelY.size() > 2 )
	{
		correlate_cols_imp<autoEnabled,PixelAccum>( src, kernelY, dst, dst_tl, option );
	}
}

/// @ingroup ImageAlgorithms
/// correlate a 2D separable variable-size kernel (kernelX and kernelY)
template <typename PixelAccum, template<typename> class Alloc, typename SrcView, typename KernelX, typename KernelY, typename DstView>
GIL_FORCEINLINE
void correlate_rows_cols_auto( const SrcView& src,
                          const KernelX& kernelX,
						  const KernelY& kernelY,
						  const DstView& dst,
						  const typename SrcView::point_t& dst_tl,
						  const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_rows_cols_imp<true,PixelAccum,Alloc,SrcView,KernelX,KernelY,DstView>( src, kernelX, kernelY, dst, dst_tl, option );
}

/// @ingroup ImageAlgorithms
/// correlate a 2D separable variable-size kernel (kernelX and kernelY)
template <typename PixelAccum, typename Alloc, typename SrcView, typename KernelX, typename KernelY, typename DstView>
GIL_FORCEINLINE
void correlate_rows_cols( const SrcView& src,
                          const KernelX& kernelX,
						  const KernelY& kernelY,
						  const DstView& dst,
						  const typename SrcView::point_t& dst_tl,
						  const convolve_boundary_option option = convolve_option_extend_zero )
{
	correlate_rows_cols_imp<false,PixelAccum,Alloc,SrcView,KernelX,KernelY,DstView>( src, kernelX, kernelY, dst, dst_tl, option );
}

}
}

#endif

