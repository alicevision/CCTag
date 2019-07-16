/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>

#include "cctag/Params.hpp"

namespace cctag {

struct FrameParam
{
    float cannyThrLow_x_256;
    float cannyThrHigh_x_256;
    float ratioVoting;
    int   thrGradientMagInVote;
    int   distSearch;
    int   minVotesToSelectCandidate;
    int   nCrowns;
    int   sampleCutLength;
    float neighbourSize;
    int   gridNSample;
    int   numCutsInIdentStep;

    __host__
    static void init( const cctag::Parameters& params );
};

extern __constant__ FrameParam tagParam;


} // namespace cctag

