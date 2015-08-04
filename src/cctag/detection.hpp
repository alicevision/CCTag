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
        const cv::Mat & graySrc,
        const Parameters & params,
        const cctag::CCTagMarkersBank & bank,
        const bool bDisplayEllipses = true );

void cctagDetectionFromEdges(
        CCTag::List& markers,
        std::vector<EdgePoint>& points,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const EdgePointsImage& edgesMap,
        const std::size_t frame,
        int pyramidLevel,
        double scale,
        const Parameters & params);

void createImageForVoteResultDebug(
        const cv::Mat & src,
        const WinnerMap & winners,
        std::size_t nLevel);

} // namespace cctag

#endif

