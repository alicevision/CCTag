#include "onoff.h"

#include <vector>
#include <math_constants.h>

#include <boost/thread/mutex.hpp>

#include "frame.h"
#include "assist.h"
#include "recursive_sweep.h"
#include "cctag/talk.hpp" // for DO_TALK macro

#undef  KERNEL_PRINT_ERROR_CAUSE
#undef  KERNEL_PRINT_SUCCESS_CAUSE

using namespace std;

// static boost::mutex debug_lock;

namespace popart
{

namespace linking
{
__device__
static const int xoff_select[8]    =   { 1,  1,  0, -1, -1, -1,  0,  1};

__device__
static const int yoff_select[2][8] = { { 0, -1, -1, -1,  0,  1,  1,  1},
                                       { 0,  1,  1,  1,  0, -1, -1, -1} };

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

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int4  edge_buffer[EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment
#else // DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int2  edge_buffer[EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment
#endif // DEBUG_LINKED_USE_INT4_BUFFER
__shared__ int   edge_index[2];

struct EdgeBuffer
{
    __device__ inline
    void init( int2 start )
    {
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
    __device__ inline
    int2 getPrevious( Direction d )
    {
        int2 retval;
        int  idx;
        if( d == Left ) {
            idx = edge_index[Left];
            dec( idx );
            dec( idx );
        } else {
            idx = edge_index[Right];
            inc( idx );
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
    __device__ inline
    const int2& getPrevious( Direction d )
    {
        int idx;
        if( d == Left ) {
            idx = edge_index[Left];
            dec( idx );
            dec( idx );
        } else {
            idx = edge_index[Right];
            inc( idx );
        }
        return edge_buffer[idx];
    }
#endif // DEBUG_LINKED_USE_INT4_BUFFER

#ifdef DEBUG_LINKED_USE_INT4_BUFFER
    __device__ inline
    void append( Direction d, int2 val, short2 grad, int offset, float angle )
    {
        if( d == Left ) {
            const int idx = edge_index[Left];
            edge_buffer[idx].x = val.x;
            edge_buffer[idx].y = val.y;
            edge_buffer[idx].z = offset;
            edge_buffer[idx].w = rintf(angle);
            assert( idx != edge_index[Right] );
            inc( edge_index[Left] );
        } else {
            assert( edge_index[Left] != edge_index[Right] );
            dec( edge_index[Right] );
            const int idx = edge_index[Right];
            edge_buffer[idx].x = val.x;
            edge_buffer[idx].y = val.y;
            edge_buffer[idx].z = offset;
            edge_buffer[idx].w = rintf(angle);
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

    __device__ bool isAlreadyIn( const int2& pt ) {
        int limit = edge_index[Left];
        int i     = 0;
        while( i != limit ) {
            if( edge_buffer[i].x == pt.x && edge_buffer[i].y == pt.y ) {
                return true;
            }
            inc( i );
        }
        limit = edge_index[Right];
        if( limit != 0 ) {
            i = 0;
            do {
                dec( i );
                if( edge_buffer[i].x == pt.x && edge_buffer[i].y == pt.y ) {
                    return true;
                }
            } while( i != limit );
        }
        return false;
    }

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

#if 0
__global__
void verify( int  num,
             int* d_ring_index,
             int* d_ring_sort_keys,
             cv::cuda::PtrStepSzInt2 d_ring_output )
{
    int ct;
    printf("Direct:\n");
    ct = 0;
    for( int i = 0; i<num; i++ ) {
        int key = d_ring_sort_keys[i];
        if( key == 0 ) { ct++; continue; }
        printf("Line %d, counter %d\n", i, key );
    }
    printf("... and %d zeros\n", ct );
    printf("Sorted:\n");
    ct = 0;
    for( int i = 0; i<num; i++ ) {
        int offset = d_ring_index[i];
        int key    = d_ring_sort_keys[i];
        if( key == 0 ) { ct++; continue; }
        printf("Line %d, index %d counter %d\n", i, offset, key );
    }
    printf("... and %d zeros\n", ct );
}
#endif

__global__
void edge_init_sort_keys( FrameMetaPtr meta,
                          int*         d_ring_sort_keys,
                          int*         d_ring_index )
{
    int limit = meta.ring_counter_max();
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < limit ) {
        d_ring_index[idx]     = idx;
        d_ring_sort_keys[idx] = 0;
    }
}

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
void edge_linking_seed( FrameMetaPtr                 meta,
                        const TriplePoint*           p,
                        DevEdgeList<TriplePoint>     voters,
                        cv::cuda::PtrStepSz32s       edgepoint_index_table, // coord->triplepoint
                        cv::cuda::PtrStepSzb         edges,
                        cv::cuda::PtrStepSz16s       d_dx,
                        cv::cuda::PtrStepSz16s       d_dy,
                        cv::cuda::PtrStepSzInt2      d_ring_output,
                        int                          my_ring_index,
                        int*                         d_ring_index,
                        int*                         d_ring_sort_keys,
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
        tcg.x = d_dx.ptr(tcc.y)[tcc.x];
        tcg.y = d_dy.ptr(tcc.y)[tcc.x];
        float atanval = atan2f( tcg.x, tcg.y );

        float angle = fmodf( atanval + 2.0f * CUDART_PI_F, 2.0f * CUDART_PI_F );

        phi.push_back( angle ); // thread 0 stores and all get the angle diff

        int shifting = rintf( ( (angle + CUDART_PI_F / 4.0f)
                              / (2.0f * CUDART_PI_F) ) * 8.0f ) - 1;

        // int j = threadIdx.x; // 0..7
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

        // bring threads back into sync
        bool any_point_found = __any( point_found );

        if( any_point_found ) {
            if( point_found ) {
                point_found = not buf.isAlreadyIn( new_point );
            }

            any_point_found = __any( point_found );
        }

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
        if( point_found == false ) {
            j = 8;
        }
        j = min( j, __shfl_down( j, 4 ) );
        j = min( j, __shfl_down( j, 2 ) );
        j = min( j, __shfl_down( j, 1 ) );
        j =         __shfl     ( j, 0 );
        assert( j < 8 );
#endif

        found = LOW_FLOW;

        float winnerSize = 0.0f;
        if( threadIdx.x == j ) {
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
                buf.append( direction, new_point, tcg, shifting, atanval*100 );
#else // DEBUG_LINKED_USE_INT4_BUFFER
                buf.append( direction, new_point );
#endif // DEBUG_LINKED_USE_INT4_BUFFER
                int idx = edgepoint_index_table.ptr(new_point.y)[new_point.x];
                if( idx > 0 ) {
                    // ptr can be any seed or voter candidate, and its _winnerSize
                    // may be 0
                    assert( idx < meta.list_size_voters() );
                    TriplePoint* ptr = &voters.ptr[idx];
                    winnerSize = ptr->_winnerSize;
                }

                // we collect votes after the IF-block, using a reduce
                found = FOUND_NEXT;
            } else {
                found = CONVEXITY_LOST;
            }
        } // end of asynchronous block
        assert( found == LOW_FLOW || found == FOUND_NEXT || found == CONVEXITY_LOST );

        __threadfence_block();

        // both FOUND_NEXT and CONVEXITY_LOST are > LOW_FLOW
        found = (StopCondition)max( (int)found, __shfl_down( (int)found, 4 ) );
        found = (StopCondition)max( (int)found, __shfl_down( (int)found, 2 ) );
        found = (StopCondition)max( (int)found, __shfl_down( (int)found, 1 ) );
        found = (StopCondition)                 __shfl     ( (int)found, 0 );

        assert( found == FOUND_NEXT || found == CONVEXITY_LOST );

        if( found == FOUND_NEXT ) {
            found      = STILL_SEARCHING;
            // only the thread going into the if() is not null
            winnerSize = winnerSize + __shfl_down( winnerSize, 1 );
            winnerSize = winnerSize + __shfl_down( winnerSize, 2 );
            winnerSize = winnerSize + __shfl_down( winnerSize, 4 );
            winnerSize =              __shfl     ( winnerSize, 0 );
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

        ++i;

        averageVote = averageVoteCollect / buf.size();
    } // while

    if( found == STILL_SEARCHING && averageVote < param_averageVoteMin ) {
        found = VOTE_LOW;
    }

    if( threadIdx.x == 0 )
    {
        if( (i == EDGE_LINKING_MAX_EDGE_LENGTH) || (found == CONVEXITY_LOST) || (found == FULL_CIRCLE) || ( found == EDGE_NOT_FOUND ) ) {
            int convexEdgeSegmentSize = buf.size();
            if (convexEdgeSegmentSize > param_windowSizeOnInnerEllipticSegment) {
                int write_index = d_ring_index[ my_ring_index ];

                // int write_index = atomicAdd( &meta.ring_counter(), 1 );
                if( write_index <= meta.ring_counter_max() ) {
#ifdef KERNEL_PRINT_SUCCESS_CAUSE
                    const char* c;
                    if( i == EDGE_LINKING_MAX_EDGE_LENGTH ) {
                        c = "max length";
                    } if( found == CONVEXITY_LOST ) {
                        c = "conv lost";
                    } if( found == EDGE_NOT_FOUND ) {
                        c = "no edge";
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

                    d_ring_sort_keys[ write_index ] = convexEdgeSegmentSize;
                }
#ifdef KERNEL_PRINT_ERROR_CAUSE
                else {
                    printf("From (%d,%d): %d (average vote %f) - skip, max number of arcs reached (%d)\n", p->coord.x, p->coord.y, i, averageVote, meta.ring_counter_max() );
                }
#endif // KERNEL_PRINT_ERROR_CAUSE
            } else {
                int write_index = d_ring_index[ my_ring_index ];
                if( write_index <= meta.ring_counter_max() ) {
                    d_ring_sort_keys[ write_index ] = 0;
                }
#ifdef KERNEL_PRINT_ERROR_CAUSE
                int d = param_windowSizeOnInnerEllipticSegment;
                printf("From (%d,%d): %d (average vote %f) - skip, edge segment size %d <= %d\n", p->coord.x, p->coord.y, i, averageVote, convexEdgeSegmentSize, d );
#endif // KERNEL_PRINT_ERROR_CAUSE
            }
        } else {
            int write_index = d_ring_index[ my_ring_index ];
            if( write_index <= meta.ring_counter_max() ) {
                d_ring_sort_keys[ write_index ] = 0;
            }
#ifdef KERNEL_PRINT_ERROR_CAUSE
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
#endif // KERNEL_PRINT_ERROR_CAUSE
        }
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
void edge_linking( FrameMetaPtr                 meta,
                   DevEdgeList<TriplePoint>     voters,
                   int                          starting_offset,
                   DevEdgeList<int>             seed_indices,
                   cv::cuda::PtrStepSz32s       edgepoint_index_table,
                   cv::cuda::PtrStepSzb         edges,
                   cv::cuda::PtrStepSz16s       d_dx,
                   cv::cuda::PtrStepSz16s       d_dy,
                   cv::cuda::PtrStepSzInt2      d_ring_output,
                   int                          d_ring_start_index,
                   int*                         d_ring_index,
                   int*                         d_ring_sort_keys,
                   size_t param_windowSizeOnInnerEllipticSegment,
                   float  param_averageVoteMin )
{
    const int limit = meta.ring_counter_max();

    const int offset = starting_offset + blockIdx.x;

    // The first seed index is always invalid
    // Should never happen
    if( offset == 0 ) return;

    // The arc spanned by this seed does not fit into the array in this round
    if( offset + d_ring_start_index > limit ) return;

    const int my_ring_index = d_ring_start_index + offset;

    int idx = seed_indices.ptr[offset];
    if( idx >= meta.list_size_voters() ) return;

    TriplePoint* p = &voters.ptr[idx];

    edge_linking_seed( meta,
                       p,
                       voters,
                       edgepoint_index_table,
                       edges,
                       d_dx,
                       d_dy,
                       d_ring_output,
                       my_ring_index,
                       d_ring_index,
                       d_ring_sort_keys,
                       param_windowSizeOnInnerEllipticSegment,
                       param_averageVoteMin );
}

__global__
void edge_exclude_seed( FrameMetaPtr                 meta,
                        int                          round,
                        DevEdgeList<TriplePoint>     voters,
                        int                          starting_offset,
                        DevEdgeList<int>             seed_indices,
                        cv::cuda::PtrStepSzInt2      d_ring_output,
                        int                          my_ring_index,
                        int*                         d_ring_index,
                        int*                         d_ring_sort_keys )
{
    int offset = blockIdx.x;
    int search_in_offset = offset - round;

    if( search_in_offset < 0 ) return;
    if( d_ring_sort_keys[offset] == 0 ) return;
    if( d_ring_sort_keys[search_in_offset] == 0 ) return;

    int seed_index           = d_ring_index[offset];
    int search_in_seed_index = d_ring_index[search_in_offset];
    int pt_index             = seed_indices.ptr[seed_index];

    TriplePoint* p = &voters.ptr[ pt_index ];
    int2 coord = p->coord;

    bool same = false;
    int i = threadIdx.x;
    while( __any( i<EDGE_LINKING_MAX_RING_BUFFER_SIZE ) ) {
        bool tst = false;
        if( i < EDGE_LINKING_MAX_RING_BUFFER_SIZE ) {
            tst = ( coord.x == d_ring_output.ptr(search_in_seed_index)[i].x )
               && ( coord.y == d_ring_output.ptr(search_in_seed_index)[i].y );
        }
        if( __any( tst ) ) {
            same = true;
            break;
        }
        i += blockDim.x;
    }

    if( same ) {
        for( int i=0; i<EDGE_LINKING_MAX_RING_BUFFER_SIZE; i++ ) {
            d_ring_output.ptr(seed_index)[i].x = 0;
            d_ring_output.ptr(seed_index)[i].y = 0;
        }
        if( threadIdx.x == 0 ) {
            d_ring_sort_keys[offset] = 0;
        }
    }
}

__global__
void edge_count_block( FrameMetaPtr meta, int* d_ring_sort_keys )
{
    int offset = threadIdx.x;
    while( not __any( offset >= EDGE_LINKING_MAX_ARCS ) )
    {
        bool isNull = ( d_ring_sort_keys[offset] == 0 );
        if( __any( isNull ) ) {
            int ct = isNull ? offset : EDGE_LINKING_MAX_ARCS;
            ct = min( ct, __shfl_down( ct, 16 ) );
            ct = min( ct, __shfl_down( ct,  8 ) );
            ct = min( ct, __shfl_down( ct,  4 ) );
            ct = min( ct, __shfl_down( ct,  2 ) );
            ct = min( ct, __shfl_down( ct,  1 ) );
            if( threadIdx.x == 0 ) {
                meta.ring_counter() = ct;
            }
            return;
        }
        offset += 32;
    }
}

}; // namespace linking

__host__
void Frame::applyLinkExcludeSeeds( int starting_offset )
{
    int ring_counter;
    _meta.fromDevice( Ring_counter, ring_counter, _stream );
    cudaStreamSynchronize( _stream );

    dim3 block( 32, 1, 1 );
    dim3 grid( ring_counter, 1, 1 );

    for( int loop = 1; loop < ring_counter; loop++ ) {
        linking::edge_exclude_seed
            <<<grid,block,0,_stream>>>
            ( _meta,
              loop,
              _voters.dev,
              starting_offset,
              _inner_points.dev,
              _d_ring_output,
              0,
              _d_ring_index,
              _d_ring_sort_keys );
    }
}

__host__
void Frame::applyLinkSortByLength( )
{
    cudaStreamSynchronize( _stream );

    cudaError_t err;

    int* counter1 = (int*)_d_ring_sort_keys;
    int* counter2 = (int*)_d_intermediate.ptr(0);
    cub::DoubleBuffer<int> keys( counter1, counter2 );

    int* index1   = (int*)_d_ring_index;
    int* index2   = (int*)_d_intermediate.ptr(_d_intermediate.rows/2);
    cub::DoubleBuffer<int> values( index1, index2 );

    if( _d_intermediate.rows/2 * _d_intermediate.step < EDGE_LINKING_MAX_ARCS * sizeof(int) ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    _d_intermediate is too small for sorting arc segment indices" << std::endl;
        exit( -1 );
    }

    void*  assist_buffer = (void*)_d_map.data;
    size_t assist_buffer_sz = 0;

    err = cub::DeviceRadixSort::SortPairsDescending(
                0,
                assist_buffer_sz,
                keys,
                values,
                EDGE_LINKING_MAX_ARCS,
                0,
                sizeof(int)*8,
                _stream,
                DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs init step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }
    if( assist_buffer_sz >= _d_map.step * _d_map.rows ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs requires too much intermediate memory. Crashing." << std::endl;
        exit( -1 );
    }

    err = cub::DeviceRadixSort::SortPairsDescending(
                assist_buffer,
                assist_buffer_sz,
                keys,
                values,
                EDGE_LINKING_MAX_ARCS,
                0,
                sizeof(int)*8,
                _stream,
                DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs compute step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }

    POP_CHK_CALL_IFSYNC;

    if( keys.Current() != counter1 ) {
        err = cudaMemcpyAsync( counter1, counter2, EDGE_LINKING_MAX_ARCS*sizeof(int), cudaMemcpyDeviceToDevice, _stream );
    }
    if( values.Current() != index1 ) {
        err = cudaMemcpyAsync( index1, index2, EDGE_LINKING_MAX_ARCS*sizeof(int), cudaMemcpyDeviceToDevice, _stream );
    }

    POP_CHK_CALL_IFSYNC;

    dim3 ct_block( 32, 1, 1 );
    dim3 ct_grid( 1, 1, 1 );

    linking::edge_count_block
        <<<ct_grid,ct_block,0,_stream>>>
        ( _meta,
          _d_ring_sort_keys );
}

__host__
void Frame::applyLink( const cctag::Parameters& params )
{
    if( params._windowSizeOnInnerEllipticSegment > EDGE_LINKING_MAX_RING_BUFFER_SIZE ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter ring buffer size is "
             << EDGE_LINKING_MAX_RING_BUFFER_SIZE << "," << endl
             << "    parameter file wants " << params._windowSizeOnInnerEllipticSegment << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

    if( _inner_points.host.size <= 0 ) {
        // We have note found any seed, return
        return;
    }

#if 0
    POP_CUDA_SYNC( _stream );
    std::cerr << "  Searching arcs from " << _inner_points.host.size << " seeds" << std::endl;
    std::cerr << "  Parameters: _windowSizeOnInnerEllipticSegment="
         << params._windowSizeOnInnerEllipticSegment << std::endl
         << "              _averageVoteMin=" << params._averageVoteMin << std::endl;
#endif

    /* Both init steps should be done in another stream, earlier. No reason to do
     * this synchronously.
     */
    _meta.toDevice( Ring_counter, 0, _stream );

    POP_CUDA_MEMSET_ASYNC( _d_ring_output.data, 0, _d_ring_output.step*_d_ring_output.rows, _stream );


    dim3 init_block( 32, 1, 1 );
    dim3 init_grid ( grid_divide( EDGE_LINKING_MAX_ARCS, 32 ), 1, 1 );

    linking::edge_init_sort_keys
        <<<init_grid,init_block,0,_stream>>>
        ( _meta,
          _d_ring_sort_keys,
          _d_ring_index );

    // cudaStreamSynchronize( _stream );
    // linking::verify<<<1,1,0,_stream>>> ( EDGE_LINKING_MAX_ARCS, _d_ring_index, _d_ring_sort_keys, _d_ring_output );

    /* Seeds have an index in the _inner_points list.
     * For each of those seeds, mark their coordinate with a label.
     * This label is their index in the _inner_points list, because
     * it is a unique int strictly > 0
     */
    dim3 link_block( 8, 1, 1 );
    dim3 link_grid ( _inner_points.host.size, 1, 1 );

    linking::edge_linking
        <<<link_grid,link_block,0,_stream>>>
        ( _meta,
          _voters.dev,
          1,                   // we start with seed 1 because seed 0 is always invalid
          _inner_points.dev,
          _vote._d_edgepoint_index_table,
          _d_edges,
          _d_dx,
          _d_dy,
          _d_ring_output,
          0,
          _d_ring_index,
          _d_ring_sort_keys,
          params._windowSizeOnInnerEllipticSegment,
          params._averageVoteMin );

    // linking::verify<<<1,1,0,_stream>>> ( EDGE_LINKING_MAX_ARCS, _d_ring_index, _d_ring_sort_keys, _d_ring_output );

    applyLinkSortByLength( );

    applyLinkExcludeSeeds( 1 ); // we start with seed 1 because seed 0 is always invalid

    applyLinkSortByLength( );

    // linking::verify<<<1,1,0,_stream>>> ( EDGE_LINKING_MAX_ARCS, _d_ring_index, _d_ring_sort_keys, _d_ring_output );

    int ring_counter;
    _meta.fromDevice( Ring_counter, ring_counter, _stream );
    POP_CUDA_SYNC( _stream );

    POP_CUDA_MEMCPY_2D_ASYNC( _h_ring_output.data, _h_ring_output.step,
                              _d_ring_output.data, _d_ring_output.step,
                              _d_ring_output.cols*sizeof(cv::cuda::PtrStepInt2_base_t),
                              _d_ring_output.rows,
                              cudaMemcpyDeviceToHost,
                              _stream );

    POP_CHK_CALL_IFSYNC;

#if 0
// #ifndef NDEBUG
    std::cerr << "  Found arcs from " << ring_counter << " seeds" << std::endl;
    for( int y = 0; y<_h_ring_output.rows; y++ ) {

        if( _h_ring_output.ptr(y)[0].x == 0 &&
            _h_ring_output.ptr(y)[0].y == 0 ) continue;

        std::cerr << "Row " << y << ": ";
        for( int x = 0; x<_h_ring_output.cols; x++ ) {
            std::cerr << "(" << _h_ring_output.ptr(y)[x].x
                      << "," << _h_ring_output.ptr(y)[x].y << ") ";

            if( _h_ring_output.ptr(y)[x].x == 0 &&
                _h_ring_output.ptr(y)[x].y == 0 ) break;
        }
        std::cerr << std::endl;
    }
#endif // NDEBUG
}
}; // namespace popart

