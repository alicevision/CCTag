#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// #include <thrust/system/cuda/detail/cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

#ifdef USE_SEPARABLE_COMPILATION
__host__
void Frame::applyVote( const cctag::Parameters& )
{
    // everything that was in here is called frame_desc.cu by
    // descent::dp_caller when USE_SEPARABLE_COMPILATION is
    // used
}
#else // not USE_SEPARABLE_COMPILATION
__host__
void Frame::applyVote( const cctag::Parameters& params )
{
    bool success;

    /* For every potential outer ring point in _voters,
     * check whether it votes for a point, and if yes:
     * (a) add that point to _seed_indices, 
     * (b) store the index that point (value it edgepoint_index_table)
     *     in the outer ring point's TriplePoint structure
     * (c) store the flow value there as well.
     */
    success = applyVoteConstructLine( );

#ifndef NDEBUG
    _voters.copySizeFromDevice( _stream, EdgeListCont );
    _vote._seed_indices.copySizeFromDevice( _stream, EdgeListWait );
    cerr << "after constructline, voters: " << _voters.host.size
         << " votes: " << _vote._seed_indices.host.size << endl;
#endif

    if( success ) {
        /* Apply sort and uniq to the list of potential inner ring
         * points in _seed_indices. Store the result in _seed_indices_2.
         */
        success = applyVoteSortUniqNoDP( params );
#ifndef NDEBUG
        _vote._seed_indices_2.copySizeFromDevice( _stream, EdgeListWait );
        cerr << "after sort/uniq, votes: " << _vote._seed_indices_2.host.size << endl;
#endif

        /* For all potential inner points in _seed_indices_2,
         * count the number of voters and compute the average
         * flow size. Annotate inner points.
         */
        applyVoteEval( params );
#ifndef NDEBUG
        _vote._seed_indices_2.copySizeFromDevice( _stream, EdgeListWait );
        cerr << "after eval, votes: " << _vote._seed_indices_2.host.size << endl;
#endif

        /* For all inner points in _seed_indices_2, check if
         * average flow size exceeds threshold, store all successful
         * inner point in _seed_indices.
         */
        applyVoteIf();
#ifndef NDEBUG
        _vote._seed_indices.copySizeFromDevice( _stream, EdgeListWait );
        cerr << "after if, votes: " << _vote._seed_indices.host.size << endl;
#else
        _vote._seed_indices.copySizeFromDevice( _stream, EdgeListCont );
#endif

        /* would it be better to remove unused voters from the chaincoords ? */
    }

    if( not success ) {
        _vote._seed_indices.host.size       = 0;
        _voters.host.size = 0;
        return;
    }

    applyVoteDownload( );
}
#endif // not USE_SEPARABLE_COMPILATION

#ifndef NDEBUG
__device__
void debug_inner_test_consistency( const char*                    origin,
                                   int                            p_idx,
                                   const TriplePoint*             p,
                                   cv::cuda::PtrStepSz32s         edgepoint_index_table,
                                   const DevEdgeList<TriplePoint> voters )
{
    if( p == 0 ) {
        printf("%s Impossible bug, initialized from memory address\n", origin);
        assert( 0 );
    }

    if( outOfBounds( p->coord, edgepoint_index_table ) ) {
        printf("%s Index (%d,%d) does not fit into coord lookup tables\n", origin, p->coord.x, p->coord.y );
        assert( 0 );
    }

    int idx = edgepoint_index_table.ptr(p->coord.y)[p->coord.x];
    if( idx < 0 || idx >= voters.Size() ) {
        printf("%s Looked up index (coord) is out of bounds\n", origin);
        assert( 0 );
    }

    if( idx != p_idx ) {
        printf("%s Looked up index %d is not identical to input index %d\n", origin, idx, p_idx);
        assert( 0 );
    }

    if( outOfBounds( p->descending.befor, edgepoint_index_table ) ) {
        printf("%s Before coordinates (%d,%d) do not fit into lookup tables\n", origin, p->descending.befor.x, p->descending.befor.y );
        assert( 0 );
    }

    if( outOfBounds( p->descending.after, edgepoint_index_table ) ) {
        printf("%s After coordinates (%d,%d) for [%d]=(%d,%d) do not fit into lookup tables\n", origin, p->descending.after.x, p->descending.after.y, p_idx, p->coord.x, p->coord.y );
        assert( 0 );
    }
}
#endif // NDEBUG

} // namespace popart

