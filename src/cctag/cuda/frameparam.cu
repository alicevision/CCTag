/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "frameparam.h"
#include "frame.h"
#include "debug_macros.hpp"

namespace cctag {

__constant__ FrameParam tagParam;

static bool tagParamInitialized = false;

__host__
void FrameParam::init( const cctag::Parameters& params )
{
    if( tagParamInitialized ) {
        return;
    }

    tagParamInitialized = true;

    if( params._nCrowns > RESERVE_MEM_MAX_CROWNS ) {
        std::cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << std::endl
             << "    static maximum of parameter crowns is "
             << RESERVE_MEM_MAX_CROWNS
             << ", parameter file wants " << params._nCrowns << std::endl
             << "    edit " << __FILE__ << " and recompile" << std::endl
             << std::endl;
    }

    FrameParam p;
    p.cannyThrLow_x_256         = params._cannyThrLow * 256.0f;
    p.cannyThrHigh_x_256        = params._cannyThrHigh * 256.0f;
    p.ratioVoting               = params._ratioVoting;
    p.thrGradientMagInVote      = params._thrGradientMagInVote;
    p.distSearch                = params._distSearch;
    p.minVotesToSelectCandidate = params._minVotesToSelectCandidate;
    p.nCrowns                   = params._nCrowns;
    p.neighbourSize             = params._imagedCenterNeighbourSize;
    p.gridNSample               = params._imagedCenterNGridSample;
    p.sampleCutLength           = params._sampleCutLength;
    p.numCutsInIdentStep        = params._numCutsInIdentStep;

    cudaError_t err;
    err = cudaMemcpyToSymbol( tagParam, // _d_symbol_ptr,
                              &p,
                              sizeof( FrameParam ),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Could not copy CCTag params to device symbol tagParam" );
}

} // namespace cctag

