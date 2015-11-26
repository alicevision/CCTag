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

struct NumVotersIsGreaterEqual
{
    DevEdgeList<TriplePoint> _array;
    int                      _compare;

    __host__ __device__
    __forceinline__
    NumVotersIsGreaterEqual( int compare, DevEdgeList<TriplePoint> _d_array )
        : _compare(compare)
        , _array( _d_array )
    {}

    __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_array.ptr[a]._winnerSize >= _compare);
    }
};

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
inline
void debug_inner_test_consistency( int                            p_idx,
                                   const TriplePoint*             p,
                                   cv::cuda::PtrStepSz32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> chained_edgecoords )
{
    if( p == 0 ) {
        printf("Impossible bug, initialized from memory address\n");
        assert( 0 );
    }

    if( outOfBounds( p->coord, edgepoint_index_table ) ) {
        printf("Index (%d,%d) does not fit into coord lookup tables\n", p->coord.x, p->coord.y );
        assert( 0 );
    }

    int idx = edgepoint_index_table.ptr(p->coord.y)[p->coord.x];
    if( idx < 0 || idx >= chained_edgecoords.Size() ) {
        printf("Looked up index (coord) is out of bounds\n");
        assert( 0 );
    }

    if( idx != p_idx ) {
        printf("Looked up index %d is not identical to input index %d\n", idx, p_idx);
        assert( 0 );
    }

    if( outOfBounds( p->descending.befor, edgepoint_index_table ) ) {
        printf("Before coordinations (%d,%d) do not fit into lookup tables\n", p->descending.befor.x, p->descending.befor.y );
        assert( 0 );
    }

    if( outOfBounds( p->descending.after, edgepoint_index_table ) ) {
        printf("After coordinations (%d,%d) do not fit into lookup tables\n", p->descending.after.x, p->descending.after.y );
        assert( 0 );
    }
}
#endif // NDEBUG

}; // namespace popart

