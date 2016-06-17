#include "onoff.h"

#include <cuda_runtime.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "cuda/cmp_list.h"

#include "frame.h"

#include <tbb/tbb.h>

namespace popart {

using namespace std;

bool Frame::applyExport( cctag::EdgePointCollection& out_edges,
                         std::vector<cctag::EdgePoint*>& out_seedlist)
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    int vote_sz = _voters.host.size;
    int all_sz  = _edgepoints.host.size;

    assert( out_seedlist.size() == 0 );
    assert( vote_sz <= all_sz );

    if( vote_sz <= 0 ) {
        // no voting happened, no need for edge linking,
        // so no need for copying anything
        // cerr << "Leave " << __FUNCTION__ << " (1)" << endl;
        return false;
    }

    /* Block 1
     * - Create a vector of EdgePoint structures with one entry for
     *   every edge point found on the GPU.
     * - Initialize each EdgePoint with (x,y) and (dx,dy) info.
     * - Create an image-sized lookup array, where the (x,y)-pos holds
     *   a pointer to an EdgePoint when it exists, or 0 if not.
     */

#ifdef SORT_ALL_EDGECOORDS_IN_EXPORT
    /* remove a bit of randomness by sorting the list of edge point
     * that is stored in the 1D array of int2 _edgepoints
     *
     * NOTE: move sorting to the GPU !!!
     */
    int2cmp v_comp;
    std::sort( _edgepoints.host.ptr,
               _edgepoints.host.ptr+all_sz,
               v_comp );
#endif // SORT_ALL_EDGECOORDS_IN_EXPORT

    for(int i = 0; i < all_sz; ++i) {
          const short2& pt = _edgepoints.host.ptr[i]._coord;
          const int16_t dx = _h_dx.ptr(pt.y)[pt.x];
          const int16_t dy = _h_dy.ptr(pt.y)[pt.x];
          out_edges.add_point(pt.x, pt.y, dx, dy);
    }

    /* Block 2
     * Copying the linkage info for all edge points that voted for an inner
     * point.
     * Perhaps important: in contrast to the CPU implementation, the GPU
     * has not removed those voters whose potential inner point did not reach
     * the threshold of voters.
     * Consequently, we have allocated memory for all of those above, and
     * we are copying linkage information for them in this block.
     */

    for( int i=1; i<vote_sz; i++ ) {
        int voter_index = _voters.host.ptr[i];
        const CudaEdgePoint& pt = _edgepoints.host.ptr[voter_index];
        cctag::EdgePoint* ep = out_edges(pt._coord.x,pt._coord.y);
        if( ep == 0 ) {
            cerr << __FILE__ << ":" << __LINE__ << ": "
                 << "Error: found a vote winner (" << pt._coord.x << "," << pt._coord.y << ")"
                 << " that is not an edge point." << endl;
            // cerr << "Leave " << __FUNCTION__ << " (2)" << endl;
            return false;
        }
        assert( ep->gradient()(0) == (double)pt._grad.x );
        assert( ep->gradient()(1) == (double)pt._grad.y );

        if( pt._dev_after != 0 ) {
            int after_index = _voters.host.ptr[pt._dev_after];
            const CudaEdgePoint& p = _edgepoints.host.ptr[after_index];
            cctag::EdgePoint* n = out_edges(p._coord.x, p._coord.y);
            if( n >= 0 )
                out_edges.set_after(ep, out_edges(n));
        }
        if( pt._dev_befor != 0 ) {
            int befor_index = _voters.host.ptr[pt._dev_befor];
            const CudaEdgePoint& p = _edgepoints.host.ptr[befor_index];
            cctag::EdgePoint* n = out_edges(p._coord.x, p._coord.y);
            if( n >= 0 )
                out_edges.set_before(ep, out_edges(n));
        }

        ep->_flowLength = pt._flowLength;
        ep->_isMax      = pt._dev_winnerSize;
    }

    /* Block 3
     * Seeds are candidate points that may be on an inner ellipse.
     * Every seed in _inner_points has received enough votes to reach
     * the threshold.
     * We sort them by number of votes first, and coordinates in case of collision.
     * We copy all of them into the host-side list.
     *
     * NOTE: move sorting to the GPU !!!
     */
#if 0
    vote_index_sort sort_by_votes_and_coords( _voters.host );
    std::sort( _inner_points.host.ptr,
               _inner_points.host.ptr + _inner_points.host.size,
               sort_by_votes_and_coords );
#endif

    // NVCC handles the std::list<...>() construct. GCC does not. Keeping alternative code.

    // std::list<cctag::EdgePoint*> empty_list;
    out_seedlist.resize( _inner_points.host.size );

    for( int i=0; i<_inner_points.host.size; i++ ) {
        int seedidx = _inner_points.host.ptr[i];

        /* NOTE: this is probably incorrect */
        int voter_index = _voters.host.ptr[seedidx];
        const CudaEdgePoint& pt = _edgepoints.host.ptr[voter_index];

        cctag::EdgePoint* ep = out_edges(pt._coord.x, pt._coord.y);
        out_seedlist[i] = ep;
    }

    /* Block 4
     * We walk through all voters, and if they have cast a vote for
     * a candidate inner point, they are added into the list for
     * that inner point.
     */
#ifndef WITH_CUDA
    std::vector<std::vector<int>> voter_lists(out_edges.get_point_count());
#else
    std::vector<std::vector<int> > voter_lists(out_edges.get_point_count());
#endif
    for( int i=1; i<vote_sz; i++ ) {
        int voter_index = _voters.host.ptr[i];
        const CudaEdgePoint& pt = _edgepoints.host.ptr[voter_index];

        const int vote = _voting_for.host.ptr[i];

        if( vote != 0 ) {
            int voter_index = _voters.host.ptr[ vote ];
            const CudaEdgePoint& point = _edgepoints.host.ptr[voter_index];
            int potential_seed = out_edges(out_edges(point._coord.x, point._coord.y));
            voter_lists[potential_seed].push_back(out_edges(out_edges(pt._coord.x,pt._coord.y)));
        }
    }
    out_edges.create_voter_lists(voter_lists);


    // cerr << "Leave " << __FUNCTION__ << " (ok)" << endl;
    return true;
}

cv::Mat* Frame::getPlane( ) const
{
    cv::Mat* ptr = new cv::Mat( _h_plane.rows, _h_plane.cols,
                                CV_8UC1,
                                _h_plane.data, _h_plane.step);
    return ptr;
}

cv::Mat* Frame::getDx( ) const
{
    cv::Mat* ptr = new cv::Mat( _h_dx.rows, _h_dx.cols,
                                CV_16SC1,
                                _h_dx.data, _h_dx.step);
    return ptr;
}

cv::Mat* Frame::getDy( ) const
{
    cv::Mat* ptr = new cv::Mat( _h_dy.rows, _h_dy.cols,
                                CV_16SC1,
                                _h_dy.data, _h_dy.step);
    return ptr;
}

cv::Mat* Frame::getMag( ) const
{
    cv::Mat* ptr = new cv::Mat( _h_mag.rows, _h_mag.cols,
                                CV_32SC1,
                                _h_mag.data, _h_mag.step);
    return ptr;
}

cv::Mat* Frame::getEdges( ) const
{
    cv::Mat* ptr = new cv::Mat( _h_edges.rows, _h_edges.cols,
                                CV_8UC1,
                                _h_edges.data, _h_edges.step);
    return ptr;
}

}; // namespace popart

