/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "onoff.h"

namespace cv {
    namespace cuda {
        using PtrStepSz16s       = PtrStepSz<int16_t>;
        using PtrStepSz32u       = PtrStepSz<uint32_t>;
        using PtrStepSz32s       = PtrStepSz<int32_t>;
        using PtrStepSzb4        = PtrStepSz<uchar4>;

        using PtrStep16s         = PtrStep<int16_t>;
        using PtrStep32u         = PtrStep<uint32_t>;
        using PtrStep32s         = PtrStep<int32_t>;
        using PtrStepb4          = PtrStep<uchar4>;

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
        using PtrStepSzInt2      = PtrStepSz<int4>;
        using PtrStepInt2        = PtrStep<int4>;
        using PtrStepInt2_base_t = int4;
#else // DEBUG_LINKED_USE_INT4_BUFFER
        using PtrStepSzInt2      = PtrStepSz<int2>;
        using PtrStepInt2        = PtrStep<int2>;
        using PtrStepInt2_base_t = int2;
#endif // DEBUG_LINKED_USE_INT4_BUFFER
    }
};

namespace cctag {

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

} // namespace cctag

