#include "frameparam.h"
#include "frame.h"
#include "debug_macros.hpp"

namespace popart {

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
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter crowns is "
             << RESERVE_MEM_MAX_CROWNS
             << ", parameter file wants " << params._nCrowns << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
    }

    FrameParam p;
    p.cannyThrLow_x_256         = params._cannyThrLow * 256.0f;
    p.cannyThrHigh_x_256        = params._cannyThrHigh * 256.0f;
    p.ratioVoting               = params._ratioVoting;
    p.thrGradientMagInVote      = params._thrGradientMagInVote;
    p.distSearch                = params._distSearch;
    p.minVotesToSelectCandidate = params._minVotesToSelectCandidate;
    p.nCrowns                   = params._nCrowns;

    cudaError_t err;
    err = cudaMemcpyToSymbol( tagParam, // _d_symbol_ptr,
                              &p,
                              sizeof( FrameParam ),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Could not copy CCTag params to device symbol tagParam" );
}

} // namespace popart

