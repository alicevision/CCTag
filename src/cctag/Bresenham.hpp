/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_BRENSENHAM_HPP_
#define _CCTAG_BRENSENHAM_HPP_

#include <cctag/geometry/Point.hpp>

#include <cstddef>

#include <opencv/cv.h>

#include "Types.hpp"

namespace cctag {

class EdgePoint;

/** @brief descent in the gradient direction from a maximum gradient point (magnitude sense) to another one.
 *
 */

EdgePoint* gradientDirectionDescent(
  const EdgePointCollection& canny,
  const EdgePoint& p,
  const int dir,
  const std::size_t nmax,
  const cv::Mat & dx, 
  const cv::Mat & dy,
  int thrGradient);

} // namespace cctag

#endif
