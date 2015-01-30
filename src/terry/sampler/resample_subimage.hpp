#ifndef _TERRY_SAMPLER_RESAMPLE_SUBIMAGE_HPP_
#define _TERRY_SAMPLER_RESAMPLE_SUBIMAGE_HPP_

#include "resample.hpp"

#include <terry/geometry/affine.hpp>


namespace terry {

/**
 * @brief Copy into the destination a rotated rectangular region from the source, rescaling it to fit into the destination
 *
 * Extract into dst the rotated bounds [src_min..src_max] rotated at 'angle' from the source view 'src'
 * The source coordinates are in the coordinate space of the source image
 * Note that the views could also be variants (i.e. any_image_view)
 */
template <typename Sampler, typename SrcMetaView, typename DstMetaView> 
void resample_subimage( const SrcMetaView& src, const DstMetaView& dst,
                         double src_min_x, double src_min_y,
                         double src_max_x, double src_max_y,
                         double angle, const Sampler& sampler=Sampler() )
{
    const double src_width  = std::max<double>( src_max_x - src_min_x - 1, 1 );
    const double src_height = std::max<double>( src_max_y - src_min_y - 1, 1 );
    const double dst_width  = std::max<double>( dst.width()-1, 1 );
    const double dst_height = std::max<double>( dst.height()-1, 1 );

    const matrix3x2<double> mat = 
        matrix3x2<double>::get_translate( -dst_width/2.0, -dst_height/2.0 ) * 
        matrix3x2<double>::get_scale( src_width / dst_width, src_height / dst_height ) *
        matrix3x2<double>::get_rotate( -angle ) *
        matrix3x2<double>::get_translate( src_min_x + src_width/2.0, src_min_y + src_height/2.0 );

    resample_pixels( src, dst, mat, sampler );
}

/**
 * @brief Copy the source view into the destination, scaling to fit.
 */
template <typename Sampler, typename SrcMetaView, typename DstMetaView> 
void resize_view( const SrcMetaView& src, const DstMetaView& dst, const Sampler& sampler=Sampler() )
{
    resample_subimage( src, dst, 0, 0, src.width(), src.height(), 0, sampler );
}


}

#endif

