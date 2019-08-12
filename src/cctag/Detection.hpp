/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_CCTAG_DETECTION_HPP_
#define VISION_CCTAG_DETECTION_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/Types.hpp>
#include <cctag/Params.hpp>
#include <cctag/Plane.hpp>
#include <cctag/utils/LogTime.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace cctag {

class EdgePoint;
class EdgePointImage;

/**
 * @brief Perform the CCTag detection on a gray scale image. Cf. application/detection/main.cpp for example of usage.
 * 
 * @param[out] markers Detected markers. WARNING: only markers with status == 1 are valid ones. (status available via getStatus()) 
 * @param[in] pipeId Choose one of up to 3 parallel CUDA pipes
 * @param[in] frame A frame number. Can be anything (e.g. 0).
 * @param[in] imgGraySrc Gray scale input image.
 * @param[in] providedParams Contains all the parameters.
 * @param[in] bank CCTag bank.
 * @param[in] bDisplayEllipses No longer used.
 */
void cctagDetection(
        CCTag::List&      markers,
        int               pipeId,
        std::size_t       frame,
        Plane<uint8_t>&   imgGraySrc,
        const Parameters& providedParams,
        const cctag::CCTagMarkersBank & bank,
        bool              bDisplayEllipses = true,
        logtime::Mgmt*    durations = nullptr );

void cctagDetectionFromEdges(
        CCTag::List&         markers,
        EdgePointCollection& edgeCollection,
        Plane<uint8_t>       src,
        const std::vector<EdgePoint*>& seeds,
        std::size_t       frame,
        int pyramidLevel,
        float scale,
        const Parameters & providedParams,
        logtime::Mgmt* durations );

void createImageForVoteResultDebug(
        Plane<uint8_t>& src,
        std::size_t     nLevel);

} // namespace cctag

#endif

