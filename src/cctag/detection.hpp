#ifndef _ROM_VISION_CCTAG_DETECTION_HPP_
#define _ROM_VISION_CCTAG_DETECTION_HPP_

#include "CCTag.hpp"
#include "CCTagMarkersBank.hpp"
#include "types.hpp"
#include "params.hpp"

#include <cctag/frame.hpp>
//#include <rom/engine/RomConfig.hpp>

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

template<class SView>
void cctagDetection( CCTag::List& markers, const FrameId frame, const SView& svw, const cctag::Parameters & params, const bool bDisplayEllipses = true );

namespace cctag {
//template<class GradView>
//void cctagDetectionFromEdges(
//		CCTag::Vector& markers,
//		std::vector<EdgePoint>& points,
//		const boost::gil::gray8_view_t& sourceView, const GradView & cannyGradX, const GradView & cannyGradY,
//		const EdgePointsImage& edgesMap, const CCTagMarkersBank& bank,
//		const FrameId frame, const std::size_t searchDistance, const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns );
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

}

}
}
}

#include "detection.tcc"

#endif

