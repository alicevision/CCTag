#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>

#include "triple_point.h"

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
    int2*                  _d_all_edgecoord_list;
    uint32_t*              _d_all_edgecoord_list_sz;
    TriplePoint*           _d_edgelist_2;
    uint32_t*              _d_edgelist_2_sz;
    int*                   _d_edgelist_3;
    uint32_t*              _d_edgelist_3_sz;
    int*                   _d_seeds;           // copy of another pointer, don't alloc/free
    cv::cuda::PtrStepSz32s _d_next_edge_coord; // 2D plane for chaining TriplePoint coord

    int2*          _h_debug_edgelist_1;
    uint32_t       _h_debug_edgelist_1_sz;
    TriplePoint*   _h_debug_edgelist_2;
    uint32_t       _h_debug_edgelist_2_sz;
    int            _h_seed_sz;

    Voting( )
        : _h_debug_edgelist_1( 0 )
        , _h_debug_edgelist_1_sz( 0 )
        , _h_debug_edgelist_2( 0 )
        , _h_debug_edgelist_2_sz( 0 )
    { }

    ~Voting( )
    {
        release( );

        delete [] _h_debug_edgelist_1;
        delete [] _h_debug_edgelist_2;
    }

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

