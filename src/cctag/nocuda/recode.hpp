/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_RECODE_HPP_
#define _CCTAG_RECODE_HPP_

#include "cctag/Plane.hpp"

namespace cctag {
struct Parameters;
};

void recodedCanny(
  cctag::Plane<uint8_t>& imgGraySrc,
  cctag::Plane<uint8_t>& imgCanny,
  cctag::Plane<int16_t>& imgDX,
  cctag::Plane<int16_t>& imgDY,
  float low_thresh,
  float high_thresh,
  int aperture_size,
  int debug_info_level,
  const cctag::Parameters* params );
#endif

