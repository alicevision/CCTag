/*
  Copyright 2005-2007 Adobe Systems Incorporated
  Distributed under the MIT License (see accompanying file LICENSE_1_0_0.txt
  or a copy at http://opensource.adobe.com/licenses.html)
*/

/*************************************************************************************************/

#ifndef _TERRY_SAMPLER_RESAMPLE_HPP_
#define _TERRY_SAMPLER_RESAMPLE_HPP_

#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

////////////////////////////////////////////////////////////////////////////////////////
/// @file               
/// @brief support for generic image resampling
///        NOTE: The code is for example use only. It is not optimized for performance
/// @author Lubomir Bourdev and Hailin Jin \n
///         Adobe Systems Incorporated
/// @date   2005-2007 \n October 30, 2006
///
////////////////////////////////////////////////////////////////////////////////////////

namespace terry {

using namespace boost::gil;

///////////////////////////////////////////////////////////////////////////
////
////   resample_pixels: set each pixel in the destination view as the result of a sampling function over the transformed coordinates of the source view
////
///////////////////////////////////////////////////////////////////////////

template <typename MapFn> struct mapping_traits {};

/**
 * @brief Set each pixel in the destination view as the result of a sampling function over the transformed coordinates of the source view
 * @ingroup ImageAlgorithms
 *
 * The provided implementation works for 2D image views only
 */
template <typename Sampler, // Models SamplerConcept
          typename SrcView, // Models RandomAccess2DImageViewConcept
          typename DstView, // Models MutableRandomAccess2DImageViewConcept
          typename MapFn>   // Models MappingFunctionConcept
void resample_pixels( const SrcView& src_view, const DstView& dst_view, const MapFn& dst_to_src, Sampler sampler=Sampler(), const sampler::EParamFilterOutOfImage outOfImageProcess = sampler::eParamFilterOutBlack )
{
    typename DstView::point_t dst_dims = dst_view.dimensions();
    typename DstView::point_t dst_p;
    //typename mapping_traits<MapFn>::result_type src_p;

    for( dst_p.y=0; dst_p.y<dst_dims.y; ++dst_p.y )
    {
        typename DstView::x_iterator xit = dst_view.row_begin(dst_p.y);
        for( dst_p.x=0; dst_p.x<dst_dims.x; ++dst_p.x )
        {
            sample(sampler, src_view, transform(dst_to_src, dst_p), xit[dst_p.x], outOfImageProcess);
        }
    }
}

///////////////////////////////////////////////////////////////////////////
////
////   resample_pixels when one or both image views are run-time instantiated. 
////
///////////////////////////////////////////////////////////////////////////

namespace detail {
    template <typename Sampler, typename MapFn>
    struct resample_pixels_fn : public binary_operation_obj<resample_pixels_fn<Sampler,MapFn> >
    {
        MapFn  _dst_to_src;
        Sampler _sampler;
        
        resample_pixels_fn( const MapFn& dst_to_src, const Sampler& sampler )
        	: _dst_to_src(dst_to_src)
        	, _sampler(sampler)
        {}

        template <typename SrcView, typename DstView>
        GIL_FORCEINLINE void apply_compatible( const SrcView& src, const DstView& dst ) const
        {
            resample_pixels( src, dst, _dst_to_src, _sampler );
        }
    };
}

/**
 * @brief resample_pixels when the source is run-time specified
 *        If invoked on incompatible views, throws std::bad_cast()
 * @ingroup ImageAlgorithms
 */
template <typename Sampler, typename Types1, typename V2, typename MapFn>
void resample_pixels( const any_image_view<Types1>& src, const V2& dst, const MapFn& dst_to_src, Sampler sampler=Sampler() )
{
    apply_operation( src, bind(detail::resample_pixels_fn<Sampler,MapFn>(dst_to_src,sampler), _1, dst) );
}

/**
 * @brief resample_pixels when the destination is run-time specified
 *        If invoked on incompatible views, throws std::bad_cast()
 * @ingroup ImageAlgorithms
 */
template <typename Sampler, typename V1, typename Types2, typename MapFn>
void resample_pixels( const V1& src, const any_image_view<Types2>& dst, const MapFn& dst_to_src, Sampler sampler=Sampler() )
{
    apply_operation( dst, bind(detail::resample_pixels_fn<Sampler,MapFn>(dst_to_src,sampler), src, _1) );
}

/**
 * @brief resample_pixels when both the source and the destination are run-time specified
 *        If invoked on incompatible views, throws std::bad_cast()
 * @ingroup ImageAlgorithms
 */
template <typename Sampler, typename SrcTypes, typename DstTypes, typename MapFn> 
void resample_pixels( const any_image_view<SrcTypes>& src, const any_image_view<DstTypes>& dst, const MapFn& dst_to_src, Sampler sampler=Sampler() )
{
    apply_operation(src,dst,detail::resample_pixels_fn<Sampler,MapFn>(dst_to_src,sampler));
}

}

#endif

