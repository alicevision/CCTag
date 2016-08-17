/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_THINNING_HPP_
#define VISION_MARKER_THINNING_HPP_

#include <opencv/cv.h>
#include <opencv2/core/types_c.h>
#include <boost/progress.hpp>
#include <iostream>


namespace cctag {

void thin( cv::Mat & inout, cv::Mat & temp );

void imageIter( cv::Mat & in, cv::Mat & out, int* lut );

}

#endif
