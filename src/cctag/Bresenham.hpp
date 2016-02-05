#ifndef _CCTAG_BRENSENHAM_HPP_
#define _CCTAG_BRENSENHAM_HPP_

#include <cctag/Labelizer.hpp>
#include <cctag/geometry/Point.hpp>

#include <cstddef>

#include <boost/gil/typedefs.hpp>
#include <boost/multi_array.hpp>

namespace cctag {

class EdgePoint;

EdgePoint* bresenham( const boost::multi_array<EdgePoint*, 2> & canny, const EdgePoint& p1, const int dir, const std::size_t nmax );

void bresenham( const boost::gil::gray8_view_t & sView, const cctag::Point2dN<int>& p, const cctag::Point2dN<float>& dir, const std::size_t nmax );

/** @brief descent in the gradient direction from a maximum gradient point (magnitude sense) to another one.
 *
 */

EdgePoint* gradientDirectionDescent(
        const boost::multi_array<EdgePoint*, 2> & canny,
        const EdgePoint& p,
        const int dir,
        const std::size_t nmax,
        const cv::Mat & dx, 
        const cv::Mat & dy,
        int thrGradient);

} // namespace cctag

#endif
