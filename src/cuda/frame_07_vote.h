#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>
#include <cub/cub.cuh>
// #include <thrust/system/cuda/detail/cub/cub.cuh>

#include "onoff.h"

#include "triple_point.h"
#include "edge_list.h"
#include "assist.h"

namespace cv {
    namespace cuda {
        typedef PtrStepSz<int16_t>  PtrStepSz16s;
        typedef PtrStepSz<uint32_t> PtrStepSz32u;
        typedef PtrStepSz<int32_t>  PtrStepSz32s;
        typedef PtrStep<int16_t>    PtrStep16s;
        typedef PtrStep<uint32_t>   PtrStep32u;
        typedef PtrStep<int32_t>    PtrStep32s;
    }
};

namespace popart {

namespace vote {

__global__
void eval_chosen( DevEdgeList<TriplePoint> chained_edgecoords, // input-output
                  DevEdgeList<int>         seed_indices        // input
                );

}; // namespace vote

struct Voting
{
    EdgeList<int2>         _all_edgecoords;
    EdgeList<TriplePoint>  _chained_edgecoords;
    EdgeList<int>          _seed_indices;
    EdgeList<int>          _seed_indices_2;
    cv::cuda::PtrStepSz32s _d_edgepoint_index_table; // 2D plane for chaining TriplePoint coord

    void debug_download( const cctag::Parameters& params );
};

#ifndef NDEBUG
__device__
void debug_inner_test_consistency( const char*                    origin,
                                   int                            p_idx,
                                   const TriplePoint*             p,
                                   cv::cuda::PtrStepSz32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> chained_edgecoords );
#endif // NDEBUG

}; // namespace popart

