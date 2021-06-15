/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_THINNING_HPP_
#define VISION_MARKER_THINNING_HPP_

#include <opencv2/core.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace cctag {

void thin( cv::Mat & inout, cv::Mat & temp );

void imageIter( const cv::Mat & in, cv::Mat & out, const int* lut );

}

#endif
