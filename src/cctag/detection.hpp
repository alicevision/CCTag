#ifndef VISION_CCTAG_DETECTION_HPP_
#define VISION_CCTAG_DETECTION_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/types.hpp>
#include <cctag/params.hpp>
#include <cctag/frame.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/typedefs.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace cctag {

class EdgePoint;
class EdgePointImage;
     
void cctagDetection(
        CCTag::List& markers,
        const std::size_t frame,
        const boost::gil::gray8_view_t& graySrc,
        const Parameters & params,
        const cctag::CCTagMarkersBank & bank,
        const bool bDisplayEllipses = true );

void cctagDetectionFromEdges(
		CCTag::List& markers,
		std::vector<EdgePoint>& points,
		const boost::gil::gray8_view_t & sourceView,
		const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX,
		const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
		const EdgePointsImage& edgesMap,
                const std::size_t frame,
                int pyramidLevel,
                double scale,
		const Parameters & params);

void createImageForVoteResultDebug(const boost::gil::gray8_view_t & sourceView, const WinnerMap & winners);

} // namespace cctag

#endif

