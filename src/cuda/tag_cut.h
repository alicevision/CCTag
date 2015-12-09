#pragma once

#include <cuda_runtime.h>

namespace popart {

namespace identification {

struct CutStruct
{
    float2 start;     // this is moving: initially set to the approximate center of the ellipse
    float2 stop;      // not moving inside cctag::refineConicFamilyGlob
    float  beginSig;  // never changes (for all tags!)
    float  endSig;    // never changes (for all tags!)
    int    sigSize;   // never changes (for all tags!)
};

struct CutSignals
{
    uint32_t outOfBounds;
    float    sig[127];
};

} // namespace identification

}; // namespace popart

