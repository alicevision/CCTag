#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "onoff.h"

namespace cv {
    namespace cuda {
        typedef PtrStepSz<int8_t>   PtrStepSz8s;
        typedef PtrStepSz<int16_t>  PtrStepSz16s;
        typedef PtrStepSz<uint32_t> PtrStepSz32u;
        typedef PtrStepSz<int32_t>  PtrStepSz32s;
        typedef PtrStepSz<uchar4>   PtrStepSzb4;

        typedef PtrStep<int8_t>     PtrStep8s;
        typedef PtrStep<int16_t>    PtrStep16s;
        typedef PtrStep<uint32_t>   PtrStep32u;
        typedef PtrStep<int32_t>    PtrStep32s;
        typedef PtrStep<uchar4>     PtrStepb4;

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
        typedef PtrStepSz<int4>     PtrStepSzInt2;
        typedef PtrStep<int4>       PtrStepInt2;
        typedef int4                PtrStepInt2_base_t;
#else // DEBUG_LINKED_USE_INT4_BUFFER
        typedef PtrStepSz<int2>     PtrStepSzInt2;
        typedef PtrStep<int2>       PtrStepInt2;
        typedef int2                PtrStepInt2_base_t;
#endif // DEBUG_LINKED_USE_INT4_BUFFER
    }
};

namespace popart {

struct PtrStepSzbClone
{
    cv::cuda::PtrStepSzb e;

    __host__
    PtrStepSzbClone( const cv::cuda::PtrStepSzb& orig );

    __host__
    ~PtrStepSzbClone( );

private:
    PtrStepSzbClone( );
    PtrStepSzbClone( const PtrStepSzbClone& );
    PtrStepSzbClone& operator=( const PtrStepSzbClone& );
};

struct PtrStepSzbNull
{
    cv::cuda::PtrStepSzb e;

    __host__
    PtrStepSzbNull( const int width, const int height );

    __host__
    ~PtrStepSzbNull( );

private:
    PtrStepSzbNull( );
    PtrStepSzbNull( const PtrStepSzbNull& );
    PtrStepSzbNull& operator=( const PtrStepSzbNull& );
};

} // namespace popart

