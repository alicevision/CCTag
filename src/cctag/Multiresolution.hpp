/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_CCTAG_MULTIRESOLUTION_HPP_
#define VISION_CCTAG_MULTIRESOLUTION_HPP_

#include <cctag/CCTag.hpp>
#include <cctag/Params.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Circle.hpp>
#include <cctag/ImagePyramid.hpp>
#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif
#include "cctag/utils/LogTime.hpp"

#include <cstddef>
#include <cmath>
#include <vector>

namespace cctag {

struct CCTagParams
{
};

/**
 * @brief Detect all CCTag in the image using multiresolution detection.
 * 
 * @param[out] markers detected cctags
 * @param[in] srcImg
 * @param[in] frame
 * 
 */

void cctagMultiresDetection(
        CCTag::List& markers,
        const cv::Mat& imgGraySrc,
        const ImagePyramid& imagePyramid,
        std::size_t   frame,
        cctag::TagPipe*    cuda_pipe,
        const Parameters&   params,
        cctag::logtime::Mgmt* durations );

void update(CCTag::List& markers, const CCTag& markerToAdd);

} // namespace cctag


#endif

