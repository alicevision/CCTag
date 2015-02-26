#ifndef _CCTAG_VISION_CCTAG_DETECTION_HPP_
#define _CCTAG_VISION_CCTAG_DETECTION_HPP_

#include "CCTag.hpp"
#include "CCTagMarkersBank.hpp"
#include "types.hpp"
#include "params.hpp"

#include <cctag/frame.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/typedefs.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace rom {
    namespace vision {
class EdgePoint;
class EdgePointImage;
    }  // namespace vision
}  // namespace rom


namespace rom {
    namespace vision {
        namespace marker {

void cctagDetection( CCTag::List& markers, const FrameId frame, const boost::gil::rgb8_view_t& svw, const cctag::Parameters & params, const bool bDisplayEllipses = true );

            namespace cctag {

void cctagDetectionFromEdges(
		CCTag::List& markers,
		std::vector<EdgePoint>& points,
		const boost::gil::gray8_view_t & sourceView,
		const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX,
		const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
		const EdgePointsImage& edgesMap,
                const FrameId frame,
                int pyramidLevel,
                double scale,
		const cctag::Parameters & params);

void createImageForVoteResultDebug(const boost::gil::gray8_view_t & sourceView, const WinnerMap & winners);

            }  // namespace cctag
        }  // namespace marker
    }  // namespace vision
}  // namespace rom

#endif

