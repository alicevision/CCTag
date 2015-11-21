#pragma once

#include <cuda_runtime.h>

#define FRAME_META_MAX_PIPES	4
#define FRAME_META_MAX_LEVELS	8

namespace popart {

enum FrameMetaEnum
{
    Hysteresis_block_counter,         // int
    Connect_component_block_counter,  // int
    Ring_counter,                     // int
    Ring_counter_max,                 // int
    Identification_result,            // float
    Identification_resct,             // int
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    Num_nearby_points,                // int
#endif
    End_indicator
};

struct FrameMetaPtr
{
    void* _d_symbol_ptr;
    int   _pipeId;
    int   _frameId;

    /* Create this FrameMeta pointer. Only the host can do it.
     * But it is passed to the device to use all those accessor
     * functions.
     */
    __host__
    FrameMetaPtr( int pipeId, int frameId );

    /* Copy the symbol e from this pipe/frame from host to device
     */
    __host__
    void toDevice( FrameMetaEnum e, int val, cudaStream_t stream );
    __host__
    void toDevice( FrameMetaEnum e, float val, cudaStream_t stream );

    /* Copy the symbol e from this pipe/frame from device to host
     */
    __host__
    void fromDevice( FrameMetaEnum e, int& val, cudaStream_t stream );
    __host__
    void fromDevice( FrameMetaEnum e, float& val, cudaStream_t stream );

#define OFFSET_GETTER_HEADER( type, name ) \
    __device__ type& name(); \
    __device__ const type& name() const;

    OFFSET_GETTER_HEADER( int, hysteresis_block_counter )
    OFFSET_GETTER_HEADER( int, connect_component_block_counter )
    OFFSET_GETTER_HEADER( int, ring_counter )
    OFFSET_GETTER_HEADER( int, ring_counter_max )
    OFFSET_GETTER_HEADER( float, identification_result )
    OFFSET_GETTER_HEADER( int, identification_resct )
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    OFFSET_GETTER_HEADER( int, num_nearby_points )
#endif

private:
    // These default functions are actually needed for automatic
    // host-device copying:
    // FrameMetaPtr( );
    // FrameMetaPtr( const FrameMetaPtr& );
    // FrameMetaPtr& operator=( const FrameMetaPtr& );
};

} // namespace popart

