#pragma once
 
#include "cctag/params.hpp"
#include <cuda_runtime.h>

namespace popart {

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


} // namespace popart

