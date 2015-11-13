#include <cuda_runtime.h>

namespace popart {

template<typename T>
__device__
T PrefixSumWarpInclusive( T threadValue, T& total )
{
    T n;
    n = __shfl_up(threadValue,  1); if( threadIdx.x >=  1 ) threadValue += n;
    n = __shfl_up(threadValue,  2); if( threadIdx.x >=  2 ) threadValue += n;
    n = __shfl_up(threadValue,  4); if( threadIdx.x >=  4 ) threadValue += n;
    n = __shfl_up(threadValue,  8); if( threadIdx.x >=  8 ) threadValue += n;
    n = __shfl_up(threadValue, 16); if( threadIdx.x >= 16 ) threadValue += n;
    total        = __shfl(    threadValue, 31 );
    return threadValue;
}

template<typename T>
__device__
T PrefixSumWarpExclusive( T threadValue, T& total )
{
    T n;
    n = __shfl_up(threadValue,  1); if( threadIdx.x >=  1 ) threadValue += n;
    n = __shfl_up(threadValue,  2); if( threadIdx.x >=  2 ) threadValue += n;
    n = __shfl_up(threadValue,  4); if( threadIdx.x >=  4 ) threadValue += n;
    n = __shfl_up(threadValue,  8); if( threadIdx.x >=  8 ) threadValue += n;
    n = __shfl_up(threadValue, 16); if( threadIdx.x >= 16 ) threadValue += n;

    total = __shfl(    threadValue, 31 );
    n  = __shfl_up( threadValue, 1 );
    threadValue = threadIdx.x == 0 ? 0 : n;
    return threadValue;
}

} // namespace popart
