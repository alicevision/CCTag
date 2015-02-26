#ifndef _CCTAG_BRENSENHAM_HPP_
#define _CCTAG_BRENSENHAM_HPP_

//#include "EdgePoint.hpp"
#include "Labelizer.hpp"

#include <cctag/geometry/point.hpp>

#include <cstddef>

#include <boost/gil/typedefs.hpp>
#include <boost/multi_array.hpp>

namespace rom {
namespace vision {

class EdgePoint;

EdgePoint* bresenham( const boost::multi_array<EdgePoint*, 2> & canny, const EdgePoint& p1, const int dir, const std::size_t nmax );

void bresenham( const boost::gil::gray8_view_t & sView, const rom::Point2dN<int>& p, const rom::Point2dN<float>& dir, const std::size_t nmax );

/** @brief descent in the gradient direction from a maximum gradient point (magnitude sense) to another one.
 *
 */

EdgePoint* gradientDirectionDescent( const boost::multi_array<EdgePoint*, 2> & canny, const EdgePoint& p, const int dir, const std::size_t nmax, const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX, const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradY, int thrGradient);

} // namespace vision
} // namespace rom

#endif
