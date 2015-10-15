#include <cuda_runtime.h>

#include "debug_is_on_edge.h"
#include "assist.h"

namespace popart
{

using namespace std;

#ifndef NDEBUG
__global__
void debug_point_is_on_edge( cv::cuda::PtrStepSzb edge_img,
                             DevEdgeList<int2>    edge_coords )
{
    int offset = blockIdx.x * 32 + threadIdx.x;
    if( offset >= edge_coords.Size() ) return;
    int2& coord = edge_coords.ptr[offset];
    assert( coord.x > 0 );
    assert( coord.y > 0 );
    assert( coord.x < edge_img.cols );
    assert( coord.y < edge_img.rows );
    assert( edge_img.ptr(coord.y)[coord.x] == 1 );
}

__host__
void debugPointIsOnEdge( const cv::cuda::PtrStepSzb& edge_img,
                         const EdgeList<int2>&       edge_coords,
                         cudaStream_t                stream )
{
    // cerr << "  Enter " << __FUNCTION__ << endl;

    int sz;
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &sz,
                                   edge_coords.dev.getSizePtr(),
                                   sizeof(int),
                                   stream );
    POP_CUDA_SYNC( stream );
    // cerr << "    Listlength " << sz << endl;
    if( sz == 0 ) {
        // cerr << "  Leave " << __FUNCTION__ << endl;
        return;
    }

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = grid_divide( sz, 32 );
    debug_point_is_on_edge
        <<<grid,block,0,stream>>>
        ( edge_img,
          edge_coords.dev );

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( stream );
    // cerr << "  Leave " << __FUNCTION__ << endl;
}
#endif // NDEBUG

}; // namespace popart

