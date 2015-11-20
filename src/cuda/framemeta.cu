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
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    int   num_nearby_points;
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

#define HOST_DEVICE_TRANSFER_CASE( cond, val ) \
    case cond: \
        offset = (intptr_t)&frame_meta[my_meta].val - (intptr_t)frame_meta; \
        break;

__host__
void FrameMetaPtr::toDevice( FrameMetaEnum e, int val, cudaStream_t stream )
{
    const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId;
    intptr_t offset;
    switch( e ) {
    HOST_DEVICE_TRANSFER_CASE( Hysteresis_block_counter, hysteresis_block_counter )
    HOST_DEVICE_TRANSFER_CASE( Connect_component_block_counter, connect_component_block_counter )
    HOST_DEVICE_TRANSFER_CASE( Ring_counter, ring_counter )
    HOST_DEVICE_TRANSFER_CASE( Ring_counter_max, ring_counter_max )
    HOST_DEVICE_TRANSFER_CASE( Identification_resct, identification_resct )
#ifndef NDEBUG
    HOST_DEVICE_TRANSFER_CASE( Offset_tester, offset_tester )
#endif
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    HOST_DEVICE_TRANSFER_CASE( Num_nearby_points, num_nearby_points )
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
    HOST_DEVICE_TRANSFER_CASE( Identification_result, identification_result )
    case Hysteresis_block_counter:
    case Connect_component_block_counter:
    case Ring_counter:
    case Ring_counter_max:
    case Identification_resct:
#ifndef NDEBUG
    case Offset_tester:
#endif
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    case Num_nearby_points:
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
    HOST_DEVICE_TRANSFER_CASE( Hysteresis_block_counter, hysteresis_block_counter )
    HOST_DEVICE_TRANSFER_CASE( Connect_component_block_counter, connect_component_block_counter )
    HOST_DEVICE_TRANSFER_CASE( Ring_counter, ring_counter )
    HOST_DEVICE_TRANSFER_CASE( Ring_counter_max, ring_counter_max )
    HOST_DEVICE_TRANSFER_CASE( Identification_resct, identification_resct )
#ifndef NDEBUG
    HOST_DEVICE_TRANSFER_CASE( Offset_tester, offset_tester )
#endif
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    HOST_DEVICE_TRANSFER_CASE( Num_nearby_points, num_nearby_points )
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
    HOST_DEVICE_TRANSFER_CASE( Identification_result, identification_result )
    case Hysteresis_block_counter:
    case Connect_component_block_counter:
    case Ring_counter:
    case Ring_counter_max:
    case Identification_resct:
#ifndef NDEBUG
    case Offset_tester:
#endif
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    case Num_nearby_points:
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

#define OFFSET_GETTER_FUNCTION( type, name ) \
    __device__ \
    type& FrameMetaPtr::name() { \
        const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId; \
        return frame_meta[my_meta].name; \
    } \
    \
    __device__ \
    const type& FrameMetaPtr::name() const { \
        const size_t my_meta = _pipeId*FRAME_META_MAX_LEVELS+_frameId; \
        return frame_meta[my_meta].name; \
    }

OFFSET_GETTER_FUNCTION( int,   hysteresis_block_counter )
OFFSET_GETTER_FUNCTION( int,   connect_component_block_counter )
OFFSET_GETTER_FUNCTION( int,   ring_counter )
OFFSET_GETTER_FUNCTION( int,   ring_counter_max )
OFFSET_GETTER_FUNCTION( float, identification_result )
OFFSET_GETTER_FUNCTION( int,   identification_resct )
#ifndef NDEBUG
OFFSET_GETTER_FUNCTION( int,   offset_tester )
#endif
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
OFFSET_GETTER_FUNCTION( int,   num_nearby_points )
#endif

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

