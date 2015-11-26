#include "onoff.h"

#ifdef USE_SEPARABLE_COMPILATION_IN_GRADDESC
// nothing to do here
#else // USE_SEPARABLE_COMPILATION_IN_GRADDESC

#include "frame.h"

using namespace std;

namespace popart
{

__host__
void Frame::applyVoteUniqNoDP( const cctag::Parameters& params )
{
    cudaError_t err;

    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

#ifdef CUB_INIT_CALLS
	assist_buffer_sz  = 0;
	// std::cerr << "before cub::DeviceSelect::Unique(0)" << std::endl;
    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        0,
        assist_buffer_sz,
        _vote._seed_indices.dev.ptr,     // input
        _vote._seed_indices_2.dev.ptr,   // output
        _vote._seed_indices_2.dev.getSizePtr(),  // output
        _vote._seed_indices.host.size,   // input (unchanged in sort)
        _stream,
        DEBUG_CUB_FUNCTIONS );

	if( err != cudaSuccess ) {
	    std::cerr << "cub::DeviceSelect::Unique init step failed. Crashing." << std::endl;
	    std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
	    exit(-1);
	}
	if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
            std::cerr << "cub::DeviceSelect::Unique requires too much intermediate memory. Crashing." << std::endl;
	    exit( -1 );
	}
#else // not CUB_INIT_CALLS
    assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;
#endif // not CUB_INIT_CALLS

    /* Unique ensure that we check every "chosen" point only once.
     * Output is in _vote._seed_indices_2.dev
     */
    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        assist_buffer,
        assist_buffer_sz,
        _vote._seed_indices.dev.ptr,     // input
        _vote._seed_indices_2.dev.ptr,   // output
        _vote._seed_indices_2.dev.getSizePtr(),  // output
        _vote._seed_indices.host.size,   // input (unchanged in sort)
        _stream,
        DEBUG_CUB_FUNCTIONS );

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB Unique failed" );
}

} // namespace popart

#endif // USE_SEPARABLE_COMPILATION_IN_GRADDESC

