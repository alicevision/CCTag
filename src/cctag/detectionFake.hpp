#ifndef VISION_CCTAG_DETECTION_FAKE_HPP_
#define VISION_CCTAG_DETECTION_FAKE_HPP_

#include "CCTagFake.hpp"
#include "CCTagMarkersBank.hpp"
#include "types.hpp"
#include "params.hpp"

#include <rom/frame.hpp>
#include <rom/engine/RomConfig.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/typedefs.hpp>

#include <rom/boostCv/cvImage.hpp>
#include <boost/gil/image.hpp>

#include <boost/timer.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace cctag {
namespace vision {
class EdgePoint;
class EdgePointImage;
}  // namespace vision
}  // namespace cctag


namespace cctag {
namespace vision {
namespace marker {

void cctagDetection( CCTagFake::List& markers, const FrameId frame, const rgb8_view_t& svw, const cctag::Parameters & params, const bool bDisplayEllipses = true )
{
    
}

namespace cctag {
//template<class GradView>
//void cctagDetectionFromEdges(
//		CCTagFake::Vector& markers,
//		std::vector<EdgePoint>& points,
//		const boost::gil::gray8_view_t& sourceView, const GradView & cannyGradX, const GradView & cannyGradY,
//		const EdgePointsImage& edgesMap, const CCTagMarkersBank& bank,
//		const FrameId frame, const std::size_t searchDistance, const double thrVotingAngle, const double thrVotingRatio, const std::size_t numCrowns );
void cctagDetectionFromEdges(
		CCTagFake::List& markers,
		std::vector<EdgePoint>& points,
		const boost::gil::gray8_view_t & sourceView,
		const boost::gil::kth_channel_view_type<1, boost::gil::rgb32f_view_t>::type & cannyGradX,
		const boost::gil::kth_channel_view_type<2, boost::gil::rgb32f_view_t>::type & cannyGradY,
		const EdgePointsImage& edgesMap,
                const FrameId frame,
                int pyramidLevel,
                double scale,
		const cctag::Parameters & params){
    
}

} // namespace cctag
} // namespace marker
} // namespace vision
} // namespace cctag

#endif

