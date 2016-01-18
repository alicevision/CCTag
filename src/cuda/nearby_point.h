#pragma once

#include <cuda_runtime.h>

#include "geom_matrix.h"

namespace popart {

struct NearbyPoint
{
    float2 point;
    float  result;
    int    resSize;
    bool   readable;

    /* These homographies are computed once for each NearbyPoint,
     * and used for all of its Cuts. The best one must be returned.
     */
    popart::geometry::matrix3x3 mHomography;
    popart::geometry::matrix3x3 mInvHomography;
};

}; // namespace popart

