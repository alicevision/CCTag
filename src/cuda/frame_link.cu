#include <vector>
#include <math_constants.h>

#include "frame.h"

#define EDGE_NOT_FOUND -1
#define CONVEXITY_LOST -2
#define LOW_FLOW -3

using namespace std;

namespace popart
{

__constant__
static int xoff_select[8]    =   { 1,  1,  0, -1, -1, -1,  0,  1};

__constant__
static int yoff_select[2][8] = { { 0, -1, -1, -1,  0,  1,  1,  1},
                                 { 0,  1,  1,  1,  0, -1, -1, -1} };

#define EDGE_LINKING_MAX_EDGE_LENGTH    100

template<typename T>
struct ListReplacement
{
    T oops;

    __device__ ListReplacement() { }
    __device__ void push_back( T& val ) { oops = val; }
    __device__ void pop_front( ) { }
    __device__ T&   back()  { return oops; }
    __device__ T&   front() { return oops; }
    __device__ int  size() const { return 1; }
};

/**
 * @param edges         The 0/1 map of edge points
 * @param d_dx
 * @param d_dy
 * @param triplepoints  The array of points including voters and seeds
 * @param edge_indices  The array of indices of seeds in triplepoints
 * @param param_windowSizeOnInnerEllipticSegment
 * @param param_averageVoteMin
 */
__device__
void edge_linking( cv::cuda::PtrStepSzb     edges,
                   DevEdgeList<TriplePoint> triplepoints,
                   DevEdgeList<int>         edge_indices,
                   size_t param_windowSizeOnInnerEllipticSegment,
                   float  param_averageVoteMin )
{
    // pmax              = one seed

    const int offset = blockIdx.x * 32 + threadIdx.z;
    int direction = ( threadIdx.y == 0 ) ?  1   // link left
                                         : -1;  // link right

    if( offset == 0 ) return;

    int idx = edge_indices.ptr[offset];
    if( idx >= triplepoints.Size() ) return;

    TriplePoint* p = &triplepoints.ptr[idx];

    __shared__ TriplePoint* convexEdgeSegment[2 * EDGE_LINKING_MAX_EDGE_LENGTH+1];
    __shared__ int          ces_idx[2]; // 0 offset right, 1 negative offset left
    __shared__ float        averageVoteCollect[2];
    float                   averageVote =  p->_winnerSize;
    if( threadIdx.x == 0 ) {
        ces_idx[threadIdx.y] = EDGE_LINKING_MAX_EDGE_LENGTH + 1 + direction;
        averageVoteCollect[threadIdx.y] = ( threadIdx.y==0 ) ? averageVote : 0;
        if( threadIdx.y == 0 ) {
            convexEdgeSegment[EDGE_LINKING_MAX_EDGE_LENGTH+1] = p;
        }
    }
    __syncthreads();

    ListReplacement<float> phi;
    std::size_t i = 0;
    bool found = true;

    int stop = 0;


    const int* xoff = xoff_select;
    const int* yoff = yoff_select[threadIdx.y];

    while( (i < EDGE_LINKING_MAX_EDGE_LENGTH) && (found) && (averageVote >= param_averageVoteMin) )
    {
        bool skip;
        if( threadIdx.x == 0 ) {
            skip = atomicExch( &p->edgeLinking.processed, true );
        }
        skip = __shfl( skip, 0 );
        if( skip ) {
            // Another warp has processed this point or is processing this
            // point.
            // End processing.
            break;
        }

        // Angle refers to the gradient direction angle (might be optimized):
        float angle = fmodf( atan2f(p->d.x,p->d.y) + 2.0f * CUDART_PI_F, 2.0f * CUDART_PI_F );

        phi.push_back(angle);

        if (phi.size() > param_windowSizeOnInnerEllipticSegment) // TODO , 4 est un paramètre de l'algorithme, + les motifs à détecter sont importants, + la taille de la fenêtre doit être grande
        {
            phi.pop_front();
        }

        int shifting = rintf( ( (angle + CUDART_PI_F / 4.0f)
                              / (2.0f * CUDART_PI_F) ) * 8.0f ) - 1;

        // int j = threadIdx.x; // 0..7
        int j = 7 - threadIdx.x; // counting backwards, so that the winner in __ffs
                                 // is identical to winner in loop code that starts
                                 // at 0
        int sx, sy;

        if (direction == 1) {
            int off_index = ( 8 - shifting + j ) % 8;
            sx = p->coord.x + xoff[off_index];
            sy = p->coord.y + yoff[off_index];
        } else {
            int off_index = ( shifting + j ) % 8;
            sx = p->coord.x + xoff[off_index];
            sy = p->coord.y + yoff[off_index];
        }

        TriplePoint* f;
        int new_edgepoint_index;
        bool point_found = false;
        if( ( sx >= 0 && sx < edges.cols ) &&
            ( sy >= 0 && sy < edges.rows ) &&
            ( new_edgepoint_index = edges.ptr(sy)[sx] ) )
        {
            f = &triplepoints.ptr[new_edgepoint_index];
            point_found = true;
        }

        uint32_t any_point_found = __ballot( point_found );
        uint32_t computer        = __ffs( any_point_found );

        if( not any_point_found ) {
            stop  = EDGE_NOT_FOUND;
            found = false;
            break;
        }

        if( threadIdx.x == computer ) {
            if( f->edgeLinking.processed ) {
                stop  = 0;
                found = false;
                break;
            }
            //
            // The whole if/else block is identical for all j.
            // No reason to do it more than once. Astonishingly,
            // the decision to use the next point f is entirely
            // independent of its position ???
            //
            float s;
            float c;
            __sincosf( phi.back() - phi.front(), &s, &c );
            s *= direction;

            //
            // three conditions to conclude CONVEXITY_LOST
            //
            stop = ( ( ( phi.size() == param_windowSizeOnInnerEllipticSegment ) &&
                       ( s <  0.0f   ) ) ||
                     ( ( s < -0.707f ) && ( c > 0.0f ) ) ||
                     ( ( s <  0.0f   ) && ( c < 0.0f ) ) );
            
            if( not stop ) {
                // this outcome of this test does not depend on
                // threadIdx.x / j

                p->edgeLinking.processed = true;

                convexEdgeSegment[ces_idx[threadIdx.y]] = f;
                ces_idx[threadIdx.y]     += direction;
                averageVoteCollect[threadIdx.y] += f->_winnerSize;
                stop = 1;

                if( f->edgeLinking.processed ) {
                    found = false;
                }
            } else {
                stop  = CONVEXITY_LOST;
                found = false;
            }
        } // end of asynchronous block

        p     = (TriplePoint*)__shfl( (size_t)f,     computer );
        found = __shfl( found, computer );
        ++i;

        __syncthreads();

        averageVote = averageVoteCollect[0] + averageVoteCollect[1];
        int convexEdgeSegmentSize = ces_idx[0] - ces_idx[1];
        averageVote /= convexEdgeSegmentSize;
    } // while


    if( threadIdx.x == 0 && threadIdx.y == 0 )
    {
        int n = 0;
        if ((i == EDGE_LINKING_MAX_EDGE_LENGTH) || (stop == CONVEXITY_LOST)) {
            int convexEdgeSegmentSize = ces_idx[0] - ces_idx[1];
            if (convexEdgeSegmentSize > param_windowSizeOnInnerEllipticSegment) {
                for( int i=ces_idx[1]; i<ces_idx[0]; i++ ) {
                    TriplePoint* collectedP = convexEdgeSegment[i];
                    if (n == convexEdgeSegmentSize - param_windowSizeOnInnerEllipticSegment) {
                        break;
                    } else {
                        // collectedP->_processedIn = true;
                        ++n;
                    }
                }
            }
        } else if (stop == EDGE_NOT_FOUND) {
            for( int i=ces_idx[1]; i<ces_idx[0]; i++ ) {
                TriplePoint* collectedP = convexEdgeSegment[i];
                // collectedP->_processedIn = true;
            }
        }
    }
}

}; // namespace popart


