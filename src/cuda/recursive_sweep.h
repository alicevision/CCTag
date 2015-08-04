#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"

using namespace std;

namespace popart
{

namespace recursive_sweep
{

__host__
void expandEdges( cv::cuda::PtrStepSz32s& img, int* dev_counter, cudaStream_t stream );

__host__
void connectComponents( cv::cuda::PtrStepSz32s& img, int* dev_counter, cudaStream_t stream );

}; // namespace recursive_sweep
}; // namespace popart

