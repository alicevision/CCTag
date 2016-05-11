#include "onoff.h"

#include <cuda_runtime.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "cuda/cmp_list.h"

#include "frame.h"

#include <tbb/tbb.h>

namespace popart {

using namespace std;

bool Frame::applyExport( std::vector<cctag::EdgePoint>&  out_edgelist,
                         cctag::EdgePointsImage&         out_edgemap,
                         std::vector<cctag::EdgePoint*>& out_seedlist)
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    int vote_sz = _voters.host.size;
    int all_sz  = _all_edgecoords.host.size;

    assert( out_edgelist.size() == 0 );
    assert( out_edgemap.size() == 0 );
    assert( out_seedlist.size() == 0 );
    assert( winners.size() == 0 );
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
     * that is stored in the 1D array of int2 _all_edgecoords
     *
     * NOTE: move sorting to the GPU !!!
     */
    int2cmp v_comp;
    std::sort( _all_edgecoords.host.ptr,
               _all_edgecoords.host.ptr+all_sz,
               v_comp );
#endif // SORT_ALL_EDGECOORDS_IN_EXPORT

    out_edgemap.resize( boost::extents[ _d_plane.cols ][ _d_plane.rows ] );
    //std::fill( out_edgemap.origin(), out_edgemap.origin() + out_edgemap.size(), (cctag::EdgePoint*)NULL );
    memset(out_edgemap.origin(), 0, out_edgemap.size() * sizeof(cctag::EdgePoint*));

    out_edgelist.reserve(all_sz+256);
    // cctag::EdgePoint* array = new cctag::EdgePoint[ all_sz ];

    for(int i = 0; i < all_sz; ++i) {
          const short2& pt = _all_edgecoords.host.ptr[i];
          const int16_t dx = _h_dx.ptr(pt.y)[pt.x];
          const int16_t dy = _h_dy.ptr(pt.y)[pt.x];
          out_edgelist.emplace_back(cctag::EdgePoint(pt.x, pt.y, dx, dy));
          out_edgemap[pt.x][pt.y] = &out_edgelist.back();
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
        const TriplePoint& pt = _voters.host.ptr[i];
        if( pt.coord.x == 0 && pt.coord.y == 0 ) {
            static bool reported_error_once = false;
            if( not reported_error_once ) {
                cerr << __FILE__ << ":" << __LINE__ << ": "
                     << "Error: vote winners contain (0,0), which is forbidden (skip)." << endl;
                reported_error_once = true;
            }
            continue;
        }
        cctag::EdgePoint* ep = out_edgemap[pt.coord.x][pt.coord.y];
        if( ep == 0 ) {
            cerr << __FILE__ << ":" << __LINE__ << ": "
                 << "Error: found a vote winner (" << pt.coord.x << "," << pt.coord.y << ")"
                 << " that is not an edge point." << endl;
            // cerr << "Leave " << __FUNCTION__ << " (2)" << endl;
            return false;
        }
        assert( ep->_grad.getX() == (double)pt.d.x );
        assert( ep->_grad.getY() == (double)pt.d.y );

        if( pt.descending.after.x != 0 || pt.descending.after.y != 0 ) {
            cctag::EdgePoint* n = out_edgemap[pt.descending.after.x][pt.descending.after.y];
            if( n != 0 )
                ep->_after = n;
        }
        if( pt.descending.befor.x != 0 || pt.descending.befor.y != 0 ) {
            cctag::EdgePoint* n = out_edgemap[pt.descending.befor.x][pt.descending.befor.y];
            if( n != 0 )
                ep->_before = n;
        }

        ep->_flowLength = pt._flowLength;
        ep->_isMax      = pt._winnerSize;
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
    vote_index_sort sort_by_votes_and_coords( _voters.host );
    std::sort( _inner_points.host.ptr,
               _inner_points.host.ptr + _inner_points.host.size,
               sort_by_votes_and_coords );

    // NVCC handles the std::list<...>() construct. GCC does not. Keeping alternative code.

    // std::list<cctag::EdgePoint*> empty_list;
    out_seedlist.resize( _inner_points.host.size );

    for( int i=0; i<_inner_points.host.size; i++ ) {
        int seedidx = _inner_points.host.ptr[i];
        const TriplePoint& pt = _voters.host.ptr[seedidx];
        cctag::EdgePoint* ep = out_edgemap[pt.coord.x][pt.coord.y];

        out_seedlist[i] = ep;
    }

    /* Block 4
     * We walk through all voters, and if they have cast a vote for
     * a candidate inner point, they are added into the list for
     * that inner point.
     */
    for( int i=1; i<vote_sz; i++ ) {
        const TriplePoint& pt   = _voters.host.ptr[i];
        const int          vote = _v_chosen_idx.host.ptr[i];

        if( vote != 0 ) {
            const TriplePoint& point = _voters.host.ptr[ vote ];
            cctag::EdgePoint* potential_seed = out_edgemap[point.coord.x][point.coord.y];
            potential_seed->_voters.push_back(out_edgemap[pt.coord.x][pt.coord.y]);
        }
    }
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

