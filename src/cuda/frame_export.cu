#include <cuda_runtime.h>

#include "frame.h"

namespace popart {

using namespace std;

bool Frame::applyExport( std::vector<cctag::EdgePoint>&  array,
                         cctag::EdgePointsImage&         edgesMap,
                         std::vector<cctag::EdgePoint*>& seeds,
                         cctag::WinnerMap&               winners )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    int vote_sz = _voters.host.size;
    int all_sz  = _vote._all_edgecoords.host.size;

    assert( array.size() == 0 );
    assert( edgesMap.size() == 0 );
    assert( seeds.size() == 0 );
    assert( winners.size() == 0 );
    assert( vote_sz <= all_sz );

    if( vote_sz <= 0 ) {
        // no voting happened, no need for edge linking,
        // so no need for copying anything
        // cerr << "Leave " << __FUNCTION__ << endl;
        return false;
    }

    edgesMap.resize( boost::extents[ _d_plane.cols ][ _d_plane.rows ] );
    std::fill( edgesMap.origin(), edgesMap.origin() + edgesMap.size(), (cctag::EdgePoint*)NULL );

    array.resize( all_sz );
    // cctag::EdgePoint* array = new cctag::EdgePoint[ all_sz ];

    for( int i=0; i<all_sz; i++ ) {
        const int2&   pt = _vote._all_edgecoords.host.ptr[i];
        const int16_t dx = _h_dx.ptr(pt.y)[pt.x];
        const int16_t dy = _h_dy.ptr(pt.y)[pt.x];

        array[i].init( pt.x, pt.y, dx, dy );

        edgesMap[pt.x][pt.y] = &array[i];
    }
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
        cctag::EdgePoint* ep = edgesMap[pt.coord.x][pt.coord.y];
        if( ep == 0 ) {
            cerr << __FILE__ << ":" << __LINE__ << ": "
                 << "Error: found a vote winner (" << pt.coord.x << "," << pt.coord.y << ")"
                 << " that is not an edge point." << endl;
            return false;
        }
        assert( ep->_grad.getX() == (double)pt.d.x );
        assert( ep->_grad.getY() == (double)pt.d.y );

        if( pt.descending.after.x != 0 || pt.descending.after.y != 0 ) {
            cctag::EdgePoint* n = edgesMap[pt.descending.after.x][pt.descending.after.y];
            if( n != 0 )
                ep->_after = n;
        }
        if( pt.descending.befor.x != 0 || pt.descending.befor.y != 0 ) {
            cctag::EdgePoint* n = edgesMap[pt.descending.befor.x][pt.descending.befor.y];
            if( n != 0 )
                ep->_before = n;
        }

        ep->_flowLength = pt._flowLength;
        ep->_isMax      = pt._winnerSize;
    }

    // NVCC handles the std::list<...>() construct. GCC does not. Keeping alternative code.
    // std::list<cctag::EdgePoint*> empty_list;
    int* seed_array = _vote._seed_indices.host.ptr;
    for( int i=0; i<_vote._seed_indices.host.size; i++ ) {
        const TriplePoint& pt = _voters.host.ptr[ seed_array[i] ];
        cctag::EdgePoint* ep = edgesMap[pt.coord.x][pt.coord.y];
        seeds.push_back( ep );

        // winners.insert( std::pair<cctag::EdgePoint*,std::list<cctag::EdgePoint*> >( ep, empty_list ) );
        winners.insert( std::pair<cctag::EdgePoint*,
                                  std::list<cctag::EdgePoint*> >( ep, std::list<cctag::EdgePoint*>() ) );
    }

    for( int i=1; i<vote_sz; i++ ) {
        const TriplePoint& pt = _voters.host.ptr[i];

        if( pt.my_vote != 0 ) {
            const TriplePoint& point = _voters.host.ptr[ pt.my_vote ];
            cctag::EdgePoint* potential_seed = edgesMap[point.coord.x][point.coord.y];
            if( winners.find(potential_seed) != winners.end() ) {
                cctag::EdgePoint* this_voter = edgesMap[pt.coord.x][pt.coord.y];
                winners[potential_seed].push_back( this_voter );
            }
        }
    }
#ifndef NDEBUG
#if 0
    std::sort(seeds.begin(), seeds.end(), cctag::receivedMoreVoteThan);

    std::vector<cctag::EdgePoint*>::const_iterator it  = seeds.begin();
    std::vector<cctag::EdgePoint*>::const_iterator end = seeds.end();
    for( ; it!=end; it++ ) {
        cctag::EdgePoint* ep = *it;
        cout << "  " << *ep << " FL=" << ep->_flowLength
                 << " VT=" << ep->_isMax
                 << " voters=";
        std::list<cctag::EdgePoint*>::const_iterator vit  = winners[ep].begin();
        std::list<cctag::EdgePoint*>::const_iterator vend = winners[ep].end();
        for( ; vit!=vend; vit++ ) {
            cout << "(" << (*vit)->getX() << "," << (*vit)->getY() << ") ";
        }
        cout << endl;
    }
#endif
#endif // NDEBUG
    // cerr << "Leave " << __FUNCTION__ << endl;
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

