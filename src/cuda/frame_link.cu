#include <vector>
#include <math_constants.h>

#include "frame.h"
#include "assist.h"
#include "recursive_sweep.h"

#undef  ONE_THREAD_ONLY // doesn't work?
#undef  KERNEL_PRINT_ERROR_CAUSE
#define KERNEL_PRINT_SUCCESS_CAUSE

using namespace std;

namespace popart
{

namespace linking
{
#if 0
__constant__
static int xoff_select[8]    =   { 1,  1,  0, -1, -1, -1,  0,  1};

__constant__
static int yoff_select[2][8] = { { 0, -1, -1, -1,  0,  1,  1,  1},
                                 { 0,  1,  1,  1,  0, -1, -1, -1} };
#else
__device__
static const int xoff_select[8]    =   { 1,  1,  0, -1, -1, -1,  0,  1};

__device__
static const int yoff_select[2][8] = { { 0, -1, -1, -1,  0,  1,  1,  1},
                                       { 0,  1,  1,  1,  0, -1, -1, -1} };
#endif

enum Direction
{
    Left = 0,
    Right = 1
};

enum StopCondition
{
    LOW_FLOW            = -4,
    VOTE_LOW            = -3,
    CONVEXITY_LOST      = -2,
    EDGE_NOT_FOUND      = -1,
    NONE                =  0,
    STILL_SEARCHING     =  1,
    FOUND_NEXT          =  2,
    FULL_CIRCLE         =  3
};

struct RingBuffer
{
    Direction direction;
    float     ring_buffer[EDGE_LINKING_MAX_RING_BUFFER_SIZE];
    size_t    max_size;
    size_t    front_index;
    size_t    back_index;
    size_t    ct;
    float     angle_diff;

    __device__ RingBuffer( Direction dir, const size_t size ) {
        assert( size < EDGE_LINKING_MAX_RING_BUFFER_SIZE );
        direction   = dir;
        max_size    = size;
        front_index = 0;
        back_index  = 0;
        ct          = 0;
        angle_diff  = 0.0f;
    }

    __device__ inline void set_direction( Direction dir ) {
        direction = dir;
    }

    __device__ void push_back( float angle )
    {
        if( threadIdx.x == 0 ) {
            if( direction == Left ) {
                ring_buffer[back_index] = angle;
                inc( back_index );
                if( front_index == back_index )
                    inc( front_index );
                else
                    ct++;
            } else {
                dec( front_index );
                if( front_index == back_index )
                    dec( back_index );
                else
                    ct++;
                ring_buffer[front_index] = angle;
            }
            angle_diff = back() - front();
        }
        angle_diff = __shfl( angle_diff, 0 );
        ct         = __shfl( ct, 0 );
    }

    __device__ inline float diff( )
    {
        // replace in lilian's original code back()-front()
        return angle_diff;
    }

    __device__ int  inline size() const {
        return ct;
    }
private:
    __device__ inline float front( )
    {
        assert( threadIdx.x == 0 );
        assert( front_index != back_index );
        return ring_buffer[front_index];
    }

    __device__ inline float back( )
    {
        assert( threadIdx.x == 0 );
        assert( front_index != back_index );
        size_t lookup = back_index;
        dec( lookup );
        return ring_buffer[lookup];
    }

    __device__ inline void inc( size_t& idx )
    {
        assert( threadIdx.x == 0 );
        idx = ( idx >= max_size-1 ) ? 0 : idx + 1;
    }

    __device__ inline void dec( size_t& idx )
    {
        assert( threadIdx.x == 0 );
        idx = ( idx == 0 ) ? max_size-1 : idx - 1;
    }
};

#ifdef ONE_THREAD_ONLY
// include data in class
#else // ONE_THREAD_ONLY
#ifdef DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int4  edge_buffer[EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment
#else // DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int2  edge_buffer[EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment
#endif // DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int   edge_index[2];
#endif // ONE_THREAD_ONLY

struct EdgeBuffer
{
#ifdef ONE_THREAD_ONLY
    int2  edge_buffer[EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment
    int   edge_index[2];
#endif // ONE_THREAD_ONLY
    __device__ inline
    void init( int2 start )
    {
#ifdef ONE_THREAD_ONLY
#ifdef DEBUG_LINKED_USE_INT4_BUFFER
        edge_buffer[0].x  = start.x;
        edge_buffer[0].y  = start.y;
        edge_buffer[0].z  = Left;
        edge_buffer[0].w  = 0;
#else // DEBUG_LINKED_USE_INT4_BUFFER
        edge_buffer[0]    = start;
#endif // DEBUG_LINKED_USE_INT4_BUFFER
        edge_index[Left]  = 1;
        edge_index[Right] = 0;
#else // ONE_THREAD_ONLY
        if( threadIdx.x == 0 ) {
#ifdef DEBUG_LINKED_USE_INT4_BUFFER
            edge_buffer[0].x  = start.x;
            edge_buffer[0].y  = start.y;
            edge_buffer[0].z  = Left;
            edge_buffer[0].w  = 0;
#else // DEBUG_LINKED_USE_INT4_BUFFER
            edge_buffer[0]    = start;
#endif // DEBUG_LINKED_USE_INT4_BUFFER
            edge_index[Left]  = 1;
            edge_index[Right] = 0;
        }
        __syncthreads();
#endif // ONE_THREAD_ONLY
    }

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
    __device__ inline
    int2 get( Direction d )
    {
        int2 retval;
        int  idx;
        if( d == Left ) {
            idx = edge_index[Left];
            dec( idx );
        } else {
            idx = edge_index[Right];
        }
        retval.x = edge_buffer[idx].x;
        retval.y = edge_buffer[idx].y;
        return retval;
    }
#else // DEBUG_LINKED_USE_INT4_BUFFER
    __device__ inline
    const int2& get( Direction d )
    {
        int idx;
        if( d == Left ) {
            idx = edge_index[Left];
            dec( idx );
        } else {
            idx = edge_index[Right];
        }
        return edge_buffer[idx];
    }
#endif // DEBUG_LINKED_USE_INT4_BUFFER

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
    __device__ inline
    void append( Direction d, int2 val, int j )
    {
        if( d == Left ) {
            const int idx = edge_index[Left];
            edge_buffer[idx].x = val.x;
            edge_buffer[idx].y = val.y;
            edge_buffer[idx].z = j;
            edge_buffer[idx].w = idx;
            assert( idx != edge_index[Right] );
            inc( edge_index[Left] );
        } else {
            assert( edge_index[Left] != edge_index[Right] );
            dec( edge_index[Right] );
            const int idx = edge_index[Right];
            edge_buffer[idx].x = val.x;
            edge_buffer[idx].y = val.y;
            edge_buffer[idx].z = 100+j;
            edge_buffer[idx].w = idx;
        }
    }
#else // DEBUG_LINKED_USE_INT4_BUFFER
    __device__ inline
    void append( Direction d, int2 val )
    {
        if( d == Left ) {
            const int idx = edge_index[Left];
            edge_buffer[idx] = val;
            assert( idx != edge_index[Right] );
            inc( edge_index[Left] );
        } else {
            assert( edge_index[Left] != edge_index[Right] );
            dec( edge_index[Right] );
            const int idx = edge_index[Right];
            edge_buffer[idx] = val;
        }
    }
#endif // DEBUG_LINKED_USE_INT4_BUFFER

    __device__ inline
    int size() const
    {
        if( edge_index[Left] > edge_index[Right] ) {
            return edge_index[Left] - edge_index[Right];
        } else {
            return edge_index[Left] + ( EDGE_LINKING_MAX_EDGE_LENGTH - edge_index[Right] );
        }
    }

    __device__
    void copy( cv::cuda::PtrStepSzInt2 output, int idx )
    {
        int sz = size();
        assert( idx < output.rows );
        if( sz > output.cols ) {
            printf("error copying link output, columns %d entries %d\n", output.cols, size() );
            assert( sz <= output.cols );
        }
        cv::cuda::PtrStepInt2_base_t* ptr = output.ptr(idx);
        int pos=edge_index[Right];
        for( int loop=0; loop<sz; loop++ ) {
            ptr[loop] = edge_buffer[pos];
            inc(pos);
        }
    }

private:
    __device__ inline void inc( int& idx )
    {
        idx = ( idx >= EDGE_LINKING_MAX_EDGE_LENGTH-1 ) ? 0 : idx + 1;
    }

    __device__ inline void dec( int& idx )
    {
        idx = ( idx == 0 ) ? EDGE_LINKING_MAX_EDGE_LENGTH-1 : idx - 1;
    }
};

/**
 * @param p Starting seed
 * @param triplepoints          List of all edge points that were potential voters
 * @param edgepoint_index_table Map: map coordinate to triplepoints entry
 * @param edges                 Map: non-0 if edge
 * @param d_dx, d_dy            Map: gradients
 * @param param_windowSizeOnInnerEllipticSegment
 * @param param_averageVoteMin
 */
__device__
void edge_linking_seed( const TriplePoint*           p,
                        DevEdgeList<TriplePoint>     triplepoints,
                        cv::cuda::PtrStepSz32s       edgepoint_index_table, // coord->triplepoint
                        cv::cuda::PtrStepSzb         edges,
                        cv::cuda::PtrStepSz16s       d_dx,
                        cv::cuda::PtrStepSz16s       d_dy,
                        int*                         d_ring_counter,
                        int                          d_ring_counter_max,
                        cv::cuda::PtrStepSzInt2      d_ring_output,
                        const size_t param_windowSizeOnInnerEllipticSegment,
                        const float  param_averageVoteMin )
{
    Direction direction       = Left;
    Direction other_direction = Right;

    float averageVote        = p->_winnerSize;
    float averageVoteCollect = averageVote;

    EdgeBuffer buf;

    buf.init( p->coord );

    RingBuffer    phi( Left, param_windowSizeOnInnerEllipticSegment );
    size_t        i     = 1;
    StopCondition found = STILL_SEARCHING;

    // const int* xoff = xoff_select;
    // const int* yoff = yoff_select[direction];

    // int2   this_cycle_coord = p->coord;
    // short2 this_cycle_gradients;

    while( (i < EDGE_LINKING_MAX_EDGE_LENGTH) && (found==STILL_SEARCHING) && (averageVote >= param_averageVoteMin) )
    {
        // this cycle coordinates
        int2  tcc = buf.get( direction );

        if( buf.size() > 1 ) {
            // other direction coordinates
            int2 odc = buf.get( other_direction );

            if( odc.x == tcc.x && odc.y == tcc.y ) {
                // We have gone a full circle.
                // End processing.
                found = FULL_CIRCLE;
                continue;
            }
        }

        assert( not outOfBounds( tcc.x, tcc.y, d_dx ) );

        // Angle refers to the gradient direction angle (might be optimized):
        short2 tcg; // this cycle gradients
        tcg.x = d_dx.ptr(tcc.y)[tcc.y];
        tcg.y = d_dy.ptr(tcc.y)[tcc.y];
        float atanval = atan2f( tcg.x, tcg.y );

        float angle = fmodf( atanval + 2.0f * CUDART_PI_F, 2.0f * CUDART_PI_F );

        phi.push_back( angle ); // thread 0 stores and all get the angle diff

        int shifting = rintf( ( (angle + CUDART_PI_F / 4.0f)
                              / (2.0f * CUDART_PI_F) ) * 8.0f ) - 1;

        // int j = threadIdx.x; // 0..7
#ifdef ONE_THREAD_ONLY
        int j = 0;
        while( j<8 ) {
            // winner is always the lowest j that finds a point
            int  off_index = ( direction == Right ) ?  ( ( 8 - shifting + j ) % 8 )
                                                    :  (     ( shifting + j ) % 8 );
            assert( off_index >= 0 );
            assert( off_index <  8 );
            int xoffset = xoff_select[off_index];
            int yoffset = yoff_select[direction][off_index];
            int2 new_point = make_int2( tcc.x + xoffset, tcc.y + yoffset );

            if( ( new_point.x >= 0 && new_point.x < edges.cols ) &&
                ( new_point.y >= 0 && new_point.y < edges.rows ) &&
                ( edges.ptr(new_point.y)[new_point.x] > 0 ) )
            {
                // This j has found a point.

                float s;
                float c;
                __sincosf( phi.diff(), &s, &c );
                s = ( direction == Left ) ? s : -s;

                //
                // three conditions to conclude CONVEXITY_LOST
                //
                bool stop;
                stop = ( ( ( phi.size() == param_windowSizeOnInnerEllipticSegment ) &&
                         ( s <  0.0f   ) ) ||
                         ( ( s < -0.707f ) && ( c > 0.0f ) ) ||
                         ( ( s <  0.0f   ) && ( c < 0.0f ) ) );
                if( stop ) {
                    if( direction == Right ) {
                        found = CONVEXITY_LOST;
                        break;
                    } else {
                        found           = STILL_SEARCHING;
                        direction       = Right;
                        other_direction = Left;
                        phi.set_direction( Right );
                    }
                } else {
                    found = STILL_SEARCHING;
                    buf.append( direction, new_point );
                    int idx = edgepoint_index_table.ptr(new_point.y)[new_point.x];
                    if( idx > 0 ) {
                        assert( idx < triplepoints.Size() );
                        TriplePoint* ptr = &triplepoints.ptr[idx];
                        averageVoteCollect += ptr->_winnerSize;
                    }
                }

                break;
            }
            j++;
        }
        if( found == STILL_SEARCHING ) {
            // checked all 8 directions, but no point found
            if( direction == Right ) {
                found = EDGE_NOT_FOUND;
                continue;
            }
            found           = STILL_SEARCHING;
            direction       = Right;
            other_direction = Left;
            phi.set_direction( Right );
            continue;
        }
#else // not ONE_THREAD_ONLY
#if 0
        int j = 7 - threadIdx.x; // counting backwards, so that the winner in __ffs
                                 // is identical to winner in loop code that starts
                                 // at 0
#else
        int j = threadIdx.x;
#endif
        int  off_index = ( direction == Right ) ?  ( ( 8 - shifting + j ) % 8 )
                                                :  (     ( shifting + j ) % 8 );
        assert( off_index >= 0 );
        assert( off_index <  8 );
        int xoffset = xoff_select[off_index];
        int yoffset = yoff_select[direction][off_index];
        int2 new_point = make_int2( tcc.x + xoffset, tcc.y + yoffset );
        // int2 new_point = make_int2( tcc.x + xoff[off_index],
        //                             tcc.y + yoff[off_index] );

        bool point_found = false;
        if( ( new_point.x >= 0 && new_point.x < edges.cols ) &&
            ( new_point.y >= 0 && new_point.y < edges.rows ) &&
            ( edges.ptr(new_point.y)[new_point.x] > 0 ) )
        {
            // at least one of 8 threads has found an edge
            // point, and has its coordinates in new_point
            point_found = true;
        }
        uint32_t any_point_found = __ballot( point_found );

        if( not any_point_found ) {
            if( direction == Right ) {
                found = EDGE_NOT_FOUND;
                continue;
            }
            found           = STILL_SEARCHING;
            direction       = Right;
            other_direction = Left;
            phi.set_direction( Right );
            continue;
        }

#if 0
        // This direction still has points.
        // We can identify the highest threadId / lowest rotation value j
        uint32_t computer = __ffs( any_point_found ) - 1;
#else
        if( point_found == false ) j = 8;
        j = min( __shfl_xor( j, 4 );
        j = min( __shfl_xor( j, 2 );
        j = min( __shfl_xor( j, 1 );
        assert( j < 8 );
#endif

        found = LOW_FLOW;

        float winnerSize = 0.0f;
#if 0
        if( threadIdx.x == computer ) {
#else
        if( threadIdx.x == j ) {
#endif
            //
            // The whole if/else block is identical for all j.
            // No reason to do it more than once. Astonishingly,
            // the decision to use the next point f is entirely
            // independent of its position ???
            //
            float s;
            float c;
            __sincosf( phi.diff(), &s, &c );
            s = ( direction == Left ) ? s : -s;

            //
            // three conditions to conclude CONVEXITY_LOST
            //
            bool stop;

            stop = ( ( ( phi.size() == param_windowSizeOnInnerEllipticSegment ) &&
                       ( s <  0.0f   ) ) ||
                     ( ( s < -0.707f ) && ( c > 0.0f ) ) ||
                     ( ( s <  0.0f   ) && ( c < 0.0f ) ) );
            
            if( not stop ) {
#ifdef DEBUG_LINKED_USE_INT4_BUFFER
                buf.append( direction, new_point, j );
#else // DEBUG_LINKED_USE_INT4_BUFFER
                buf.append( direction, new_point );
#endif // DEBUG_LINKED_USE_INT4_BUFFER
                int idx = edgepoint_index_table.ptr(new_point.y)[new_point.x];
                if( idx > 0 ) {
                    // ptr can be any seed or voter candidate, and its _winnerSize
                    // may be 0
                    assert( idx < triplepoints.Size() );
                    TriplePoint* ptr = &triplepoints.ptr[idx];
                    winnerSize = ptr->_winnerSize;
                }

                // we collect votes after the IF-block, using a reduce
                found = FOUND_NEXT;
            } else {
                found = CONVEXITY_LOST;
            }
        } // end of asynchronous block
        assert( found == LOW_FLOW || found == FOUND_NEXT || found == CONVEXITY_LOST );

        // both FOUND_NEXT and CONVEXITY_LOST are > LOW_FLOW
        found = (StopCondition)max( (int)found, __shfl_xor( (int)found, 4 ) );
        found = (StopCondition)max( (int)found, __shfl_xor( (int)found, 2 ) );
        found = (StopCondition)max( (int)found, __shfl_xor( (int)found, 1 ) );

        assert( found == FOUND_NEXT || found == CONVEXITY_LOST );

        if( found == FOUND_NEXT ) {
            found      = STILL_SEARCHING;
            // only the thread going into the if() is not null
            winnerSize = winnerSize + __shfl_xor( winnerSize, 1 );
            winnerSize = winnerSize + __shfl_xor( winnerSize, 2 );
            winnerSize = winnerSize + __shfl_xor( winnerSize, 4 );
            averageVoteCollect += winnerSize;
        } else {
            assert( found == CONVEXITY_LOST );
            if( direction == Right ) {
                found = CONVEXITY_LOST;
                continue;
            }
            found           = STILL_SEARCHING;
            direction       = Right;
            other_direction = Left;
            phi.set_direction( Right );
        }
#endif // ONE_THREAD_ONLY

        ++i;

        averageVote = averageVoteCollect / buf.size();
    } // while

    if( found == STILL_SEARCHING && averageVote < param_averageVoteMin ) {
        found = VOTE_LOW;
    }

#ifdef ONE_THREAD_ONLY
    if( true )
#else // ONE_THREAD_ONLY
    if( threadIdx.x == 0 )
#endif // ONE_THREAD_ONLY
    {
        if( (i == EDGE_LINKING_MAX_EDGE_LENGTH) || (found == CONVEXITY_LOST) || (found == FULL_CIRCLE) ) {
            int convexEdgeSegmentSize = buf.size();
            if (convexEdgeSegmentSize > param_windowSizeOnInnerEllipticSegment) {
                int write_index = atomicAdd( d_ring_counter, 1 );
                if( write_index <= d_ring_counter_max ) {
#ifdef KERNEL_PRINT_SUCCESS_CAUSE
                    const char* c;
                    if( i == EDGE_LINKING_MAX_EDGE_LENGTH ) {
                        c = "max length";
                    } if( found == CONVEXITY_LOST ) {
                        c = "conv lost";
                    } else {
                        c = "full circle";
                    }
                    printf("From (%d,%d): %d (average vote %f) - accept (%s), edge segment size %d, write pos %d\n",
                           p->coord.x, p->coord.y,
                           i, averageVote,
                           c, convexEdgeSegmentSize,
                           write_index );
#endif // KERNEL_PRINT_SUCCESS_CAUSE
                    buf.copy( d_ring_output, write_index );
                }
#ifdef KERNEL_PRINT_ERROR_CAUSE
                else {
                    printf("From (%d,%d): %d (average vote %f) - skip, max number of arcs reached (%d)\n", p->coord.x, p->coord.y, i, averageVote, d_ring_counter_max );
                }
#endif // KERNEL_PRINT_ERROR_CAUSE
            }
#ifdef KERNEL_PRINT_ERROR_CAUSE
            else {
                int d = param_windowSizeOnInnerEllipticSegment;
                printf("From (%d,%d): %d (average vote %f) - skip, edge segment size %d <= %d\n", p->coord.x, p->coord.y, i, averageVote, convexEdgeSegmentSize, d );
            }
#endif // KERNEL_PRINT_ERROR_CAUSE
        }
#ifdef KERNEL_PRINT_ERROR_CAUSE
        else {
            const char* c;
            switch(found) {
                case LOW_FLOW : c = "LOW_FLOW"; break;
                case VOTE_LOW : c = "VOTE_LOW"; break;
                case CONVEXITY_LOST : c = "CONVEXITY_LOST"; break;
                case EDGE_NOT_FOUND : c = "EDGE_NOT_FOUND"; break;
                case NONE : c = "NONE"; break;
                case STILL_SEARCHING : c = "STILL_SEARCHING"; break;
                case FOUND_NEXT : c = "FOUND_NEXT"; break;
                case FULL_CIRCLE : c = "FULL_CIRCLE"; break;
                default: c = "UNKNOWN code"; break;
            }
            printf("From (%d,%d): %d (average vote %f) - skip, not max length, not convexity lost, but %s\n", p->coord.x, p->coord.y, i, averageVote, c );
        }
#endif // KERNEL_PRINT_ERROR_CAUSE
    }
}

/**
 * @param edges         The 0/1 map of edge points
 * @param d_dx
 * @param d_dy
 * @param triplepoints  The array of points including voters and seeds
 * @param seed_indices  The array of indices of seeds in triplepoints
 * @param d_ring_counter A frame-global counter of edge segments
 * @param d_ring_ouput   A huge buffer to hold all edge segments multiple times
 * @param param_windowSizeOnInnerEllipticSegment
 * @param param_averageVoteMin
 */
__global__
void edge_linking( DevEdgeList<TriplePoint>     triplepoints,
                   DevEdgeList<int>             seed_indices,
                   cv::cuda::PtrStepSz32s       edgepoint_index_table,
                   cv::cuda::PtrStepSzb         edges,
                   cv::cuda::PtrStepSz16s       d_dx,
                   cv::cuda::PtrStepSz16s       d_dy,
                   int*                         d_ring_counter,
                   int                          d_ring_counter_max,
                   cv::cuda::PtrStepSzInt2      d_ring_output,
                   size_t param_windowSizeOnInnerEllipticSegment,
                   float  param_averageVoteMin )
{
    const int       offset    = blockIdx.x;

    // The first seed index is always invalid
    if( offset == 0 ) return;

    int idx = seed_indices.ptr[offset];
    if( idx >= triplepoints.Size() ) return;

    TriplePoint* p = &triplepoints.ptr[idx];

    edge_linking_seed( p,
                       triplepoints,
                       edgepoint_index_table,
                       edges,
                       d_dx,
                       d_dy,
                       d_ring_counter,
                       d_ring_counter_max,
                       d_ring_output,
                       param_windowSizeOnInnerEllipticSegment,
                       param_averageVoteMin );
}

}; // namespace linking

__host__
void Frame::applyLink( const cctag::Parameters& params )
{
    cout << "Enter " << __FUNCTION__ << endl;

    if( params._windowSizeOnInnerEllipticSegment > EDGE_LINKING_MAX_RING_BUFFER_SIZE ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter ring buffer size is "
             << EDGE_LINKING_MAX_RING_BUFFER_SIZE << "," << endl
             << "    parameter file wants " << params._windowSizeOnInnerEllipticSegment << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

    if( _vote._seed_indices.host.size <= 0 ) {
        cout << "Leave " << __FUNCTION__ << endl;
        // We have note found any seed, return
        return;
    }

#ifndef NDEBUG
    POP_CUDA_SYNC( _stream );
    cout << "  Searching arcs from " << _vote._seed_indices.host.size << " seeds" << endl;
    cout << "  Parameters: _windowSizeOnInnerEllipticSegment="
         << params._windowSizeOnInnerEllipticSegment << endl
         << "              _averageVoteMin=" << params._averageVoteMin << endl;
#endif // NDEBUG

    /* Both init steps should be done in another stream, earlier. No reason to do
     * this synchronously.
     */
    POP_CUDA_SET0_ASYNC( _d_ring_counter, _stream );

    POP_CUDA_MEMSET_ASYNC( _d_ring_output.data, 0, _d_ring_output.step*_d_ring_output.rows, _stream );

    dim3 block;
    dim3 grid;

    /* Seeds have an index in the _seed_indices list.
     * For each of those seeds, mark their coordinate with a label.
     * This label is their index in the _seed_indices list, because
     * it is a unique int strictly > 0
     */
#ifdef ONE_THREAD_ONLY
    block.x = 1;
#else // ONE_THREAD_ONLY
    block.x = 8;
#endif // ONE_THREAD_ONLY
    block.y = 1;
    block.z = 1;
    grid.x  = _vote._seed_indices.host.size;
    grid.y  = 1;
    grid.z  = 1;

    linking::edge_linking
        <<<grid,block,0,_stream>>>
        ( _vote._chained_edgecoords.dev,
          _vote._seed_indices.dev,
          _vote._d_edgepoint_index_table,
          _d_edges,
          _d_dx,
          _d_dy,
          _d_ring_counter,
          _d_ring_counter_max,
          _d_ring_output,
          params._windowSizeOnInnerEllipticSegment,
          params._averageVoteMin );

    POP_CHK_CALL_IFSYNC;

    POP_CUDA_MEMCPY_2D_ASYNC( _h_ring_output.data, _h_ring_output.step,
                              _d_ring_output.data, _d_ring_output.step,
                              _d_ring_output.cols*sizeof(cv::cuda::PtrStepInt2_base_t),
                              _d_ring_output.rows,
                              cudaMemcpyDeviceToHost,
                              _stream );

    POP_CHK_CALL_IFSYNC;

#ifndef NDEBUG
    int h_ring_counter;
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &h_ring_counter, _d_ring_counter, sizeof(int), _stream );
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( _stream );
    cout << "  Found arcs from " << h_ring_counter << " seeds" << endl;
#endif // NDEBUG

    cout << "Leave " << __FUNCTION__ << endl;
}
}; // namespace popart


