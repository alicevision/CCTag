/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_CCTAG_CANNY_HPP_
#define VISION_CCTAG_CANNY_HPP_

#include <cctag/Types.hpp>

#include <opencv2/core/core.hpp>

#include <vector>


namespace cctag {

class EdgePoint;

void edgesPointsFromCanny(
        EdgePointCollection& edgeCollection,
        const cv::Mat & edges,
        const cv::Mat & dx,
        const cv::Mat & dy );

} // namespace cctag

#endif

