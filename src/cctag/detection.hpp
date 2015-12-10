#pragma once

#include <cctag/CCTag.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/types.hpp>
#include <cctag/params.hpp>
#include <cctag/frame.hpp>
#include "cctag/logtime.hpp"

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/typedefs.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace popart {
class Package;
};

namespace cctag {

class EdgePoint;
class EdgePointImage;

void cctagDetection(
        popart::Package* package,
        CCTag::List& markers,
        const std::size_t frame,
        const cv::Mat & graySrc,
        const Parameters & params,
        const cctag::CCTagMarkersBank & bank,
        const bool bDisplayEllipses = true,
        logtime::Mgmt* durations = 0 );

void cctagDetectionFromEdges(
        CCTag::List&            markers,
        std::vector<EdgePoint>& points,
        const cv::Mat&          src,
        WinnerMap&              winners,
        const std::vector<EdgePoint*>& seeds,
        const EdgePointsImage&  edgesMap,
        const std::size_t       frame,
        int pyramidLevel,
        double scale,
        const Parameters & params,
        logtime::Mgmt* durations );

void createImageForVoteResultDebug(
        const cv::Mat & src,
        const WinnerMap & winners,
        std::size_t nLevel);

} // namespace cctag

