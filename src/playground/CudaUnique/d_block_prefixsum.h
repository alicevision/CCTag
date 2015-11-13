#pragma once

#include "d_warp_prefixsum.h"
#include <cuda_runtime.h>

namespace popart {

template<typename T>
__device__
T PrefixSumBlockExclusive( const T threadValue, T& blockTotal )
{
    // pre: threadValue is an integer type
    T warpTotal;
    T returnValue = PrefixSumWarpExclusive( threadValue, warpTotal );
    // post: every thread knows its write offset relative to warp base

    __shared__ T uniqueCounter[32];

    if( threadIdx.x == 0 ) {
        uniqueCounter[threadIdx.y] = warpTotal;
        // post: uniqueCounter[i] contains total item of warp i
    }
    __syncthreads();

    size_t offset;

    if( threadIdx.y == 0 ) {
        offset = uniqueCounter[threadIdx.x];
        T n = PrefixSumWarpExclusive( offset, blockTotal );
        // blockTotal is only valid for threadIdx.y == 0
        uniqueCounter[threadIdx.x] = n;
    }
    __syncthreads();
    // post: uniqueCounter[i] contains exclusive prefix sum of warp's
    //       total item count

    offset = uniqueCounter[threadIdx.y];
    // post: every thread in a warp know the warp's base offset

    returnValue += offset;
    // post: every thread knows its write offset relative to block base

    return returnValue;
}

} // namespace popart

