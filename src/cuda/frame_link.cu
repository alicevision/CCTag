#include <vector>
#include <math_constants.h>

#include "frame.h"
#include "assist.h"
#include "recursive_sweep.h"

using namespace std;

namespace popart
{

namespace linking
{
__constant__
static int xoff_select[8]    =   { 1,  1,  0, -1, -1, -1,  0,  1};

__constant__
static int yoff_select[2][8] = { { 0, -1, -1, -1,  0,  1,  1,  1},
                                 { 0,  1,  1,  1,  0, -1, -1, -1} };

#define EDGE_LINKING_MAX_EDGE_LENGTH    100
#define MAX_RING_BUFFER_SIZE            40

enum Direction
{
    Left = 0,
    Right = 1
};

enum StopCondition
{
    LOW_FLOW        =     -3,
    CONVEXITY_LOST      = -2,
    EDGE_NOT_FOUND      = -1,
    NONE                =  0,
    STILL_SEARCHING     =  1,
    FOUND_NEXT          =  2,
    FULL_CIRCLE         =  3
};

__shared__ float ring_buffer[2][MAX_RING_BUFFER_SIZE];

struct RingBuffer
{
    Direction direction;
    size_t    max_size;
    size_t    front_index;
    size_t    back_index;
    size_t    ct;

    __device__ RingBuffer( Direction dir, size_t size ) {
        direction   = dir;
        max_size    = size;
        front_index = 0;
        back_index  = 0;
        ct          = 0;
    }

    __device__ inline void set_direction( Direction dir ) {
        direction = dir;
    }

    __device__ inline size_t inc( size_t& idx )
    {
        return ( idx == max_size-1 ) ? 0 : idx + 1;
    }

    __device__ inline size_t dec( size_t& idx )
    {
        return ( idx == 0 ) ? max_size-1 : idx - 1;
    }

    __device__ void push_back( float angle )
    {
        ring_buffer[direction][front_index].angle = angle;
        back_index = inc( back_index );
        if( front_index == back_index )
            front_index = inc( front_index );
        else
            ct++;
    }

    __device__ inline float front( )
    {
        assert( front_index != back_index );
        return ring_buffer[direction][front_index].angle;
    }

    __device__ inline float back( )
    {
        assert( front_index != back_index );
        size_t lookup = dec( back_index );
        return ring_buffer[direction][lookup].angle;
    }

    __device__ int  inline size() const {
        return ct;
    }
};

__shared__ int2  edge_buffer[2][EDGE_LINKING_MAX_EDGE_LENGTH]; // convexEdgeSegment

struct EdgeBuffer
{
    int head[2];

    __device__ inline
    void init( int2 start )
    {
        edge_buffer[Left ][0] = start;
        edge_buffer[Right][0] = start;
        head[Left ] = 0;
        head[Right] = 0;
    }

    __device__ inline
    const int2& get( Direction d )
    {
        return edge_buffer[d][head[d]];
    }

    __device__ inline
    void append( Direction d, int2 val )
    {
        head[d]++;
        edge_buffer[d][head[d]] = val;
    }

    __device__ inline
    int size() const
    {
        return head[Left] + head[Right] + 1;
    }

    __device__
    void copy( cv::cuda::PtrStepSzInt2 output, int idx )
    {
        assert( idx < output.rows );
        assert( size() < output.cols );
        int j = 0;
        int2* ptr = output.ptr(idx);
        for( int i=head[Right]; i>=0; i-- ) {
            ptr[j] = edge_buffer[Right][i];
            j++;
        }
        for( int i=1; i<=head[Left]; i++ ) {
            ptr[j] = edge_buffer[Left][i];
            j++;
        }
    }
};

/**
 * @param p Starting seed
 * @param triplepoints          List of all edge points that were potential voters
 * @param edgepoint_index_table Map: map coordinate to triplepoints entry
 * @param edges                 Map: non-0 if edge
 * @param param_windowSizeOnInnerEllipticSegment
 * @param param_averageVoteMin
 */
__device__
void edge_linking_seed( const TriplePoint*           p,
                        DevEdgeList<TriplePoint>     triplepoints,
                        cv::cuda::PtrStepSz32s       edgepoint_index_table, // coord->triplepoint
                        cv::cuda::PtrStepSzb         edges,
                        int*                         d_ring_counter,
                        cv::cuda::PtrStepSzInt2      d_ring_output,
                        size_t param_windowSizeOnInnerEllipticSegment,
                        float  param_averageVoteMin )
{
    const Direction direction       = Left;
    const Direction other_direction = Right;

    float averageVote        = p->_winnerSize;
    float averageVoteCollect = averageVote;

    EdgeBuffer buf;

    if( threadIdx.x == 0 ) {
        buf.init( p->coord );
    }
    __syncthreads();

    RingBuffer<float> phi( Left, param_windowSizeOnInnerEllipticSegment );
    size_t        i     = 0;
    StopCondition found = STILL_SEARCHING;
    // int           stop  = 0;

    const int* xoff = xoff_select;
    const int* yoff = yoff_select[direction];

    // int2   this_cycle_coord = p->coord;
    // short2 this_cycle_gradients;

    while( (i < EDGE_LINKING_MAX_EDGE_LENGTH) && (found==STILL_SEARCHING) && (averageVote >= param_averageVoteMin) )
    {
        // this cycle coordinates
        const int2&  tcc = buf.get( direction );

        if( i > 0 ) {
            // other direction coordinates
            const int2& odc = buf.get( other_direction );

            if( odc.x == tcc.x && odc.y == tcc.y ) {
                // We have gone a full circle.
                // End processing.
                found = FULL_CIRCLE;
                continue;
            }
        }

        float angle;

        if( threadIdx.x == 0 ) {
            // this cycle gradients
            short2 tcg;

            // Angle refers to the gradient direction angle (might be optimized):
            tcg.x = _d_dx.ptr(tcc.y)[tcc.y];
            tcg.y = _d_dy.ptr(tcc.y)[tcc.y];
            float atanval = atan2f( tcg.x, tcg.y );

            angle = fmodf( atanval + 2.0f * CUDART_PI_F, 2.0f * CUDART_PI_F );

            phi.push_back( angle );
        }
        angle = __shfl( angle, 0 );

        int shifting = rintf( ( (angle + CUDART_PI_F / 4.0f)
                              / (2.0f * CUDART_PI_F) ) * 8.0f ) - 1;

        // int j = threadIdx.x; // 0..7
        int j = 7 - threadIdx.x; // counting backwards, so that the winner in __ffs
                                 // is identical to winner in loop code that starts
                                 // at 0
        int  off_index = ( direction == Right ) ?  ( ( 8 - shifting + j ) % 8 )
                                                :  (     ( shifting + j ) % 8 );
        int2 new_point = make_int2( tcc.x + xoff[off_index],
                                    tcc.y + yoff[off_index] );

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
            ring_buffer.set_direction( Right );
            continue;
        }

        // This direction still has points.
        // We can identify the highest threadId / lowest rotation value j
        uint32_t computer = __ffs( any_point_found );

        found = LOW_FLOW;

        float winnerSize = 0.0f;
        if( threadIdx.x == computer ) {
            //
            // The whole if/else block is identical for all j.
            // No reason to do it more than once. Astonishingly,
            // the decision to use the next point f is entirely
            // independent of its position ???
            //
            float s;
            float c;
            __sincosf( phi.back() - phi.front(), &s, &c );
            s = ( direction == Left ) ? s : -s;

            //
            // three conditions to conclude CONVEXITY_LOST
            //
            stop = ( ( ( phi.size() == param_windowSizeOnInnerEllipticSegment ) &&
                       ( s <  0.0f   ) ) ||
                     ( ( s < -0.707f ) && ( c > 0.0f ) ) ||
                     ( ( s <  0.0f   ) && ( c < 0.0f ) ) );
            
            if( not stop ) {
                buffer.append( new_point );
                int idx = edgepoint_index_table.ptr(new_point.y)[new_point.x];
                float winnerSize = 0.0f;
                if( idx > 0 ) {
                    // ptr can be any seed or voter candidate, and its _winnerSize
                    // may be 0
                    TriplePoint* ptr = &triplepoints[idx];
                    winnerSize = ptr->_winnerSize;
                }

                // we collect votes after the IF-block, using a reduce
                found = FOUND_NEXT;
            } else {
                found = CONVEXITY_LOST;
            }
        } // end of asynchronous block

        // both FOUND_NEXT and CONVEXITY_LOST are > LOW_FLOW
        found = max( found, __shfl_xor( found, 1 ) );
        found = max( found, __shfl_xor( found, 2 ) );
        found = max( found, __shfl_xor( found, 4 ) );

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
            ring_buffer.set_direction( Right );
        }

        ++i;

        averageVote = averageVoteCollect / buffer.size();
    } // while

    if( threadIdx.x == 0 )
    {
        if( (i == EDGE_LINKING_MAX_EDGE_LENGTH) || (found == CONVEXITY_LOST) ) {
            int convexEdgeSegmentSize = buffer.size();
            if (convexEdgeSegmentSize > param_windowSizeOnInnerEllipticSegment) {
                int write_index = atomicAdd( &d_ring_counter, 1 );
                if( write_index <= d_ring_counter_max ) {
                    buffer.copy( d_ring_output, write_index );
                }
            }
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
void edge_linking( DevEdgeList<TriplePoint>     triplepoints,
                   DevEdgeList<int>             seed_indices,
                   cv::cuda::PtrStepSz32s       edgepoint_index_table,
                   cv::cuda::PtrStepSzb         edges,
                   int*                         d_ring_counter,
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
                       d_ring_counter,
                       d_ring_output,
                       param_windowSizeOnInnerEllipticSegment,
                       param_averageVoteMin );
}

}; // namespace linking

__host__
void Frame::applyLink( const cctag::Parameters& params )
{
    cout << "Enter " << __FUNCTION__ << endl;

    if( param.windowSizeOnInnerEllipticSegment > MAX_RING_BUFFER_SIZE ) {
        cerr << "Error in " << __FILE__ << ":" << __LINE__ << ":" << endl
             << "    static maximum of parameter ring buffer size is "
             << MAX_RING_BUFFER_SIZE << "," << endl;
             << "    parameter file wants " << param.windowSizeOnInnerEllipticSegment << endl
             << "    edit " << __FILE__ << " and recompile" << endl
             << endl;
        exit( -1 );
    }

    if( _vote._seed_indices.host.size <= 0 ) {
        cout << "Leave " << __FUNCTION__ << endl;
        // We have note found any seed, return
        return;
    }

    /* We could re-use edges, but it has only 1-byte cells.
     * Seeds may have int-valued indices, so we need 4-byte cells.
     * _d_intermediate is a float image, and can be used.
     */
    // cv::cuda::PtrStepSz32s edge_cast( _d_intermediate );

    dim3 block;
    dim3 grid;

    /* Seeds have an index in the _seed_indices list.
     * For each of those seeds, mark their coordinate with a label.
     * This label is their index in the _seed_indices list, because
     * it is a unique int strictly > 0
     */
    block.x = 8;
    block.y = 0;
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
          _d_ring_counter,
          _d_ring_output,
          param.windowSizeOnInnerEllipticSegment,
          param.averageVoteMin );

    cout << "Leave " << __FUNCTION__ << endl;
}
}; // namespace popart


