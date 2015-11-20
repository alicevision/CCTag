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
#ifndef NDEBUG
    Offset_tester,                    // int
#endif
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

    __device__
    int&   hysteresis_block_counter();
    __device__
    int&   connect_component_block_counter();
    __device__
    int&   ring_counter();
    __device__
    int&   ring_counter_max();
    __device__
    float& identification_result();
    __device__
    int&   identification_resct();
    __device__
    const int&   hysteresis_block_counter() const;
    __device__
    const int&   connect_component_block_counter() const;
    __device__
    const int&   ring_counter() const;
    __device__
    const int&   ring_counter_max() const;
    __device__
    const float& identification_result() const;
    __device__
    const int&   identification_resct() const;

#ifndef NDEBUG
    /* Function to test whether the offsets computed are reasonable.
     * Read from device and print to cerr.
     */
    __host__
    void testOffset( cudaStream_t stream );
#endif // NDEBUG
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    __device__
    const int&   num_nearby_points() const;
#endif

private:
    // FrameMetaPtr( );
    // FrameMetaPtr( const FrameMetaPtr& );
    // FrameMetaPtr& operator=( const FrameMetaPtr& );
};

} // namespace popart

