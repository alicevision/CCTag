#include "onoff.h"

// #include <iostream>
// #include <algorithm>
// #include <limits>
#include <cuda_runtime.h>
// #include <cub/cub.cuh>
// #include <thrust/system/cuda/detail/cub/cub.cuh>
// #include <stdio.h>
// #include "debug_macros.hpp"
// #include "debug_is_on_edge.h"

#include "frame.h"
// #include "assist.h"

using namespace std;

namespace popart
{

/* After vote_eval_chosen, _voters is no longer changed
 * we can copy it to the host for edge linking
 */

__host__
void Frame::applyVoteDownload( )
{
#ifdef EDGE_LINKING_HOST_SIDE
    /* After vote_eval_chosen, _voters is no longer changed
     * we can copy it to the host for edge linking
     */
    _voters.copySizeFromDevice( _stream, EdgeListWait );
    _voters.copyDataFromDeviceAsync( _download_stream );
    _vote._seed_indices.      copyDataFromDeviceAsync( _download_stream );
#endif // EDGE_LINKING_HOST_SIDE
}

} // namespace popart

