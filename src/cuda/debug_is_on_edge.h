#include <cuda_runtime.h>

#include "frame.h"

namespace popart
{

#ifndef NDEBUG

// Called at the end of applyThinning to ensure that all coordinates
// in the edge list are also in the edge image.
// Called again in gradientDescent because the same data is suddently
// wrong?

__host__
void debugPointIsOnEdge( const cv::cuda::PtrStepSzb& edge_img,
                         const EdgeList<int2>&       edge_coords,
                         cudaStream_t          stream );

#endif // NDEBUG

}; // namespace popart

