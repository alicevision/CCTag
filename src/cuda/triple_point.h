#pragma once

#include <cuda_runtime.h>

namespace popart {

/*************************************************************
 * TriplePoint
 * A simplified version of EdgePoint in the C++ code.
 *************************************************************/
struct TriplePoint
{
    int2  coord;
    int2  befor;
    int2  after;

    // in the original code, chosen keeps list of voters
    // no possible here; we must invert this
    int   my_vote;
    float chosen_flow_length;

    int   _winnerSize;
    float _flowLength;
};

}; // namespace popart

