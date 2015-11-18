#include <iostream>

#include "onoff.h"
#include "framemeta.h"
#include "debug_macros.hpp"

// #include <cuda_runtime.h>
// #include <assert.h>

namespace popart {

/*************************************************************
 * FrameMeta
 * This structure is required for moving things such as counters
 * between host and device. It is meant to replace a memory
 * mapping function that is unreliable.
 */
struct FrameMeta
{
    int   hysteresis_block_counter;
    int   connect_component_block_counter;
    int   ring_counter;
    int   ring_counter_max;
    float identification_result;
    int   identification_resct;
#ifndef NDEBUG
    int   offset_tester;
#endif
};

__device__
FrameMeta frame_meta[ FRAME_META_MAX_PIPES * FRAME_META_MAX_LEVELS ];

__host__
FrameMetaPtr::FrameMetaPtr( int pipeId, int frameId )
    : _pipeId( pipeId )
    , _frameId( frameId )
{
    if( pipeId >= FRAME_META_MAX_PIPES ) {
	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << "Requesting more than " << FRAME_META_MAX_PIPES << " CUDA pipelines."
		  << std::endl
		  << "This requires a recompile."
		  << std::endl;
        exit( -1 );
    }
    if( frameId >= FRAME_META_MAX_LEVELS ) {
	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << "Requesting more than " << FRAME_META_MAX_LEVELS << " CUDA pipelines."
		  << std::endl
		  << "This requires a recompile."
		  << std::endl;
        exit( -1 );
    }

    cudaError_t err;
    err = cudaGetSymbolAddress( &_d_symbol_ptr, frame_meta );
    POP_CUDA_FATAL_TEST( err, "Could not recover the symbol address for FrameMetas" );
}

__host__
void FrameMetaPtr::toDevice( FrameMetaEnum e, int val, cudaStream_t stream )
{
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    intptr_t offset;
    switch( e ) {
    case Hysteresis_block_counter:
    	offset = (intptr_t)&frame_meta[my_meta].hysteresis_block_counter - (intptr_t)frame_meta;
	break;
    case Connect_component_block_counter:
    	offset = (intptr_t)&frame_meta[my_meta].connect_component_block_counter - (intptr_t)frame_meta;
	break;
    case Ring_counter:
    	offset = (intptr_t)&frame_meta[my_meta].ring_counter - (intptr_t)frame_meta;
	break;
    case Ring_counter_max:
    	offset = (intptr_t)&frame_meta[my_meta].ring_counter_max - (intptr_t)frame_meta;
	break;
    case Identification_resct:
    	offset = (intptr_t)&frame_meta[my_meta].identification_resct - (intptr_t)frame_meta;
	break;
#ifndef NDEBUG
    case Offset_tester:
    	offset = (intptr_t)&frame_meta[my_meta].offset_tester - (intptr_t)frame_meta;
	break;
#endif
    case Identification_result:
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to copy an int to FrameMeta::<float>" << std::endl
		  << "Type is incorrect." << std::endl;
	exit( -1 );
    default :
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to copy an unknown FrameMeta element." << std::endl;
	exit( -1 );
    }
    cudaError_t err;
    err = cudaMemcpyToSymbolAsync( frame_meta, // _d_symbol_ptr,
		                   &val,
				   sizeof(int),
				   offset,
				   cudaMemcpyHostToDevice,
				   stream );
    POP_CUDA_FATAL_TEST( err, "Could not copy int variable to device symbol" );
}

__host__
void FrameMetaPtr::toDevice( FrameMetaEnum e, float val, cudaStream_t stream )
{
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    intptr_t offset;
    switch( e ) {
    case Identification_result:
    	offset = (intptr_t)&frame_meta[my_meta].identification_result - (intptr_t)frame_meta;
	break;
    case Hysteresis_block_counter:
    case Connect_component_block_counter:
    case Ring_counter:
    case Ring_counter_max:
    case Identification_resct:
#ifndef NDEBUG
    case Offset_tester:
#endif
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to copy a float to a FrameMeta::<int>" << std::endl
		  << "Type is incorrect." << std::endl;
	exit( -1 );
    default :
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to copy an unknown FrameMeta element." << std::endl;
	exit( -1 );
    }
    cudaError_t err;
    err = cudaMemcpyToSymbolAsync( frame_meta, // _d_symbol_ptr,
		                   &val,
				   sizeof(int),
				   offset,
				   cudaMemcpyHostToDevice,
				   stream );
    POP_CUDA_FATAL_TEST( err, "Could not copy float variable to device symbol" );
}

__host__
void FrameMetaPtr::fromDevice( FrameMetaEnum e, int& val, cudaStream_t stream )
{
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    intptr_t offset;
    switch( e ) {
    case Hysteresis_block_counter:
    	offset = (intptr_t)&frame_meta[my_meta].hysteresis_block_counter - (intptr_t)frame_meta;
	break;
    case Connect_component_block_counter:
    	offset = (intptr_t)&frame_meta[my_meta].connect_component_block_counter - (intptr_t)frame_meta;
	break;
    case Ring_counter:
    	offset = (intptr_t)&frame_meta[my_meta].ring_counter - (intptr_t)frame_meta;
	break;
    case Ring_counter_max:
    	offset = (intptr_t)&frame_meta[my_meta].ring_counter_max - (intptr_t)frame_meta;
	break;
    case Identification_resct:
    	offset = (intptr_t)&frame_meta[my_meta].identification_resct - (intptr_t)frame_meta;
	break;
#ifndef NDEBUG
    case Offset_tester:
    	offset = (intptr_t)&frame_meta[my_meta].offset_tester - (intptr_t)frame_meta;
	std::cerr << "Trying to read value from offset " << offset << std::endl;
	break;
#endif
    case Identification_result:
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to fetch an int to FrameMeta::<float>" << std::endl
		  << "Type is incorrect." << std::endl;
	exit( -1 );
    default :
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to fetch an unknown FrameMeta element." << std::endl;
	exit( -1 );
    }
    cudaError_t err;
    err = cudaMemcpyFromSymbolAsync( &val,
		                     frame_meta, // _d_symbol_ptr,
				     sizeof(int),
				     offset,
				     cudaMemcpyDeviceToHost,
				     stream );
    POP_CUDA_FATAL_TEST( err, "Could not copy int variable from device symbol: " );
}

__host__
void FrameMetaPtr::fromDevice( FrameMetaEnum e, float& val, cudaStream_t stream )
{
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    intptr_t offset;
    switch( e ) {
    case Identification_result:
    	offset = (intptr_t)&frame_meta[my_meta].identification_result - (intptr_t)frame_meta;
	break;
    case Hysteresis_block_counter:
    case Connect_component_block_counter:
    case Ring_counter:
    case Ring_counter_max:
    case Identification_resct:
#ifndef NDEBUG
    case Offset_tester:
#endif
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to fetch a float to a FrameMeta::<int>" << std::endl
		  << "Type is incorrect." << std::endl;
	exit( -1 );
    default :
    	std::cerr << __FILE__ << ":" << __LINE__ << std::endl
		  << __FUNCTION__ << std::endl
		  << "Trying to fetch an unknown FrameMeta element." << std::endl;
	exit( -1 );
    }
    cudaError_t err;
    err = cudaMemcpyFromSymbolAsync( &val,
		                     frame_meta, // _d_symbol_ptr,
				     sizeof(int),
				     offset,
				     cudaMemcpyDeviceToHost,
				     stream );
    POP_CUDA_FATAL_TEST( err, "Could not copy float variable from device symbol: " );
}

__device__
int&   FrameMetaPtr::hysteresis_block_counter() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].hysteresis_block_counter;
}
__device__
int&   FrameMetaPtr::connect_component_block_counter() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].connect_component_block_counter;
}
__device__
int&   FrameMetaPtr::ring_counter() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].ring_counter;
}
__device__
int&   FrameMetaPtr::ring_counter_max() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].ring_counter_max;
}
__device__
float& FrameMetaPtr::identification_result() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].identification_result;
}
__device__
int&   FrameMetaPtr::identification_resct() {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].identification_resct;
}
__device__
const int&   FrameMetaPtr::hysteresis_block_counter() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].hysteresis_block_counter;
}
__device__
const int&   FrameMetaPtr::connect_component_block_counter() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].connect_component_block_counter;
}
__device__
const int&   FrameMetaPtr::ring_counter() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].ring_counter;
}
__device__
const int&   FrameMetaPtr::ring_counter_max() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].ring_counter_max;
}
__device__
const float& FrameMetaPtr::identification_result() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].identification_result;
}
__device__
const int&   FrameMetaPtr::identification_resct() const {
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    return frame_meta[my_meta].identification_resct;
}

#ifndef NDEBUG
__global__
void offset_setter( FrameMetaPtr meta )
{
    const size_t my_meta = meta._pipeId*FRAME_META_MAX_LEVELS+meta._frameId;
    int offset = (intptr_t)&frame_meta[my_meta].offset_tester - (intptr_t)frame_meta;
    frame_meta[my_meta].offset_tester = offset;
}

__host__
void FrameMetaPtr::testOffset( cudaStream_t stream )
{
    std::cerr << "Enter " << __FUNCTION__ << std::endl;
    std::cerr << "symbol address is " << std::hex << (intptr_t)_d_symbol_ptr
	      << std::dec  << std::endl;
    offset_setter
        <<<1,1,0,stream>>>
	( *this );
    int offset_value;
    fromDevice( Offset_tester, offset_value, stream );
    cudaStreamSynchronize( stream );
    std::cerr << "OFFSET TESTING" << std::endl
	      << std::endl
	      << "Offset: " << offset_value << std::endl
	      << std::endl
	      << "END OFFSET TESTING" << std::endl;
    std::cerr << "Leave " << __FUNCTION__ << std::endl;
}
#endif // NDEBUG

}; // namespace popart

