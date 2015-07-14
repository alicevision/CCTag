#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>

#include "triple_point.h"
#include "edge_list.h"

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

struct Voting
{
    EdgeList<int2>         _all_edgecoords;
    EdgeList<TriplePoint>  _chained_edgecoords;
    EdgeList<int>          _edge_indices;
    EdgeList<int>          _edge_indices_2;
    cv::cuda::PtrStepSz32s _d_edgepoint_index_table; // 2D plane for chaining TriplePoint coord

    Voting( )
    { }

    ~Voting( )
    {
        release( );
    }

    void debug_download( const cctag::Parameters& params );

    void alloc( const cctag::Parameters& params, size_t w, size_t h );
    void init( const cctag::Parameters& params, cudaStream_t stream );
    void release( );

    bool gradientDescent( const cctag::Parameters&     params,
                          const cv::cuda::PtrStepSzb   edges,
                          const cv::cuda::PtrStepSz16s d_dx,
                          const cv::cuda::PtrStepSz16s d_dy,
                          cudaStream_t                 stream );

    bool constructLine( const cctag::Parameters&     params,
                        const cv::cuda::PtrStepSz16s d_dx,
                        const cv::cuda::PtrStepSz16s d_dy,
                        cudaStream_t                 stream );
};

}; // namespace popart

