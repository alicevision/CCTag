#ifndef VISION_CCTAG_DETECTION_HPP_
#define VISION_CCTAG_DETECTION_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/Types.hpp>
#include <cctag/Params.hpp>
#include "cctag/utils/LogTime.hpp"

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
        const cv::Mat & graySrc,
        const Parameters & params,
        const cctag::CCTagMarkersBank & bank,
        const bool bDisplayEllipses = true,
        logtime::Mgmt* durations = 0 );

void cctagDetectionFromEdges(
        CCTag::List&            markers,
        EdgePointCollection& edgeCollection,
        const cv::Mat&          src,
        const std::vector<EdgePoint*>& seeds,
        const std::size_t       frame,
        int pyramidLevel,
        float scale,
        const Parameters & params,
        logtime::Mgmt* durations );

void createImageForVoteResultDebug(
        const cv::Mat & src,
        std::size_t nLevel);

} // namespace cctag

#endif

