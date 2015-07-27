#include <iostream>
#include <algorithm>
#include <limits>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <map>
#include "debug_macros.hpp"

#include "../cctag/cmdline.hpp"

#include "frame.h"
#include "debug_image.h"
#include "edge_list.h"
#include "assist.h"

#undef CHATTY_WRITE_DEBUG_PLANE

namespace popart {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

void Frame::hostDebugDownload( const cctag::Parameters& params )
{
    delete [] _h_debug_plane;
    delete [] _h_debug_smooth;
    delete [] _h_debug_dx;
    delete [] _h_debug_dy;
    delete [] _h_debug_mag;
    delete [] _h_debug_map;
    delete [] _h_debug_hyst_edges;
    delete [] _h_debug_edges;

    _h_debug_plane      = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_smooth     = new float[ getWidth() * getHeight() ];
    _h_debug_dx         = new int16_t[ getWidth() * getHeight() ];
    _h_debug_dy         = new int16_t[ getWidth() * getHeight() ];
    _h_debug_mag        = new uint32_t[ getWidth() * getHeight() ];
    _h_debug_map        = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_hyst_edges = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_edges      = new unsigned char[ getWidth() * getHeight() ];

    POP_SYNC_CHK;

    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_plane, getWidth(),
                              _d_plane.data, _d_plane.step,
                              _d_plane.cols,
                              _d_plane.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_smooth, getWidth() * sizeof(float),
                              _d_smooth.data, _d_smooth.step,
                              _d_smooth.cols * sizeof(float),
                              _d_smooth.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_dx, getWidth() * sizeof(int16_t),
                              _d_dx.data, _d_dx.step,
                              _d_dx.cols * sizeof(int16_t),
                              _d_dx.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_dy, getWidth() * sizeof(int16_t),
                              _d_dy.data, _d_dy.step,
                              _d_dy.cols * sizeof(int16_t),
                              _d_dy.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_mag, getWidth() * sizeof(uint32_t),
                              _d_mag.data, _d_mag.step,
                              _d_mag.cols * sizeof(uint32_t),
                              _d_mag.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_map, getWidth() * sizeof(uint8_t),
                              _d_map.data, _d_map.step,
                              _d_map.cols * sizeof(uint8_t),
                              _d_map.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_hyst_edges, getWidth() * sizeof(uint8_t),
                              _d_hyst_edges.data, _d_hyst_edges.step,
                              _d_hyst_edges.cols * sizeof(uint8_t),
                              _d_hyst_edges.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_edges, getWidth() * sizeof(uint8_t),
                              _d_edges.data, _d_edges.step,
                              _d_edges.cols * sizeof(uint8_t),
                              _d_edges.rows,
                              cudaMemcpyDeviceToHost, _stream );
}

void Frame::hostDebugCompare( unsigned char* pix )
{
    bool found_mistake = false;
    size_t mistake_ct = 0;

    for( int h=0; h<_d_plane.rows; h++ ) {
        for( int w=0; w<_d_plane.cols; w++ ) {
            if( pix[h*_d_plane.cols+w] != _h_debug_plane[h*_d_plane.cols+w] ) {
                mistake_ct++;
                if( found_mistake == false ) {
                    found_mistake = true;
                    cerr << "Found first error at (" << w << "," << h << "): "
                         << "orig " << pix[h*_d_plane.cols+w]
                         << "copy " << _h_debug_plane[h*_d_plane.cols+w]
                         << endl;
                }
            }
        }
    }
    if( found_mistake ) {
        cerr << "Total errors: " << mistake_ct << endl;
    } else {
        cerr << "Found no difference between original and re-downloaded frame" << endl;
    }
}

struct PtrStepSzbClone
{
    cv::cuda::PtrStepSzb e;

    PtrStepSzbClone( const cv::cuda::PtrStepSzb& orig )
        : e ( orig )
    {
        e.data = new uint8_t[ orig.rows * orig.step ];
        memcpy( e.data, orig.data, orig.rows * orig.step );
    }

    ~PtrStepSzbClone( )
    {
        delete [] e.data;
    }

private:
    PtrStepSzbClone( );
    PtrStepSzbClone( const PtrStepSzbClone& );
    PtrStepSzbClone& operator=( const PtrStepSzbClone& );
};

void Frame::writeHostDebugPlane( string filename, const cctag::Parameters& params )
{
    struct stat st = {0};

    string dir = cmdline._debugDir;
    char   dirtail = dir[ dir.size()-1 ];
    if( dirtail != '/' ) {
        filename = dir + "/" + filename;
    } else {
        filename = dir + filename;
    }

    if (stat( dir.c_str(), &st) == -1) {
        mkdir( dir.c_str(), 0700);
    }

    string s;

#ifdef DEBUG_WRITE_ORIGINAL_AS_PGM
    cv::cuda::PtrStepSzb b( getHeight(),
                            getWidth(),
                            _h_debug_plane,
                            getWidth() );
    DebugImage::writePGM( filename + ".pgm", b );
#ifdef DEBUG_WRITE_ORIGINAL_AS_ASCII
    DebugImage::writeASCII( filename + "-img-ascii.txt", b );
#endif // DEBUG_WRITE_ORIGINAL_AS_ASCII
#endif // WRITE_ORIGINAL_AS_PGM

#ifdef DEBUG_WRITE_GAUSSIAN_AS_PGM
    cv::cuda::PtrStepSzf smooth( getHeight(),
                                 getWidth(),
                                 _h_debug_smooth,
                                 getWidth()*sizeof(float) );
    DebugImage::writePGMscaled( filename + "-gauss.pgm", smooth );
#ifdef DEBUG_WRITE_GAUSSIAN_AS_ASCII
    DebugImage::writeASCII( filename + "-gauss-ascii.txt", smooth );
#endif // DEBUG_WRITE_GAUSSIAN_AS_ASCII
#endif // DEBUG_WRITE_GAUSSIAN_AS_PGM

#ifdef DEBUG_WRITE_DX_AS_PGM
    cv::cuda::PtrStepSz16s dx( getHeight(),
                               getWidth(),
                               _h_debug_dx,
                               getWidth()*sizeof(int16_t) );
    DebugImage::writePGMscaled( filename + "-dx.pgm", dx );
#ifdef DEBUG_WRITE_DX_AS_ASCII
    DebugImage::writeASCII( filename + "-dx-ascii.txt", dx );
#endif // DEBUG_WRITE_DX_AS_ASCII
#endif // DEBUG_WRITE_DX_AS_PGM

#ifdef DEBUG_WRITE_DY_AS_PGM
    cv::cuda::PtrStepSz16s dy( getHeight(),
                               getWidth(),
                               _h_debug_dy,
                               getWidth()*sizeof(int16_t) );
    DebugImage::writePGMscaled( filename + "-dy.pgm", dy );
#ifdef DEBUG_WRITE_DY_AS_ASCII
    DebugImage::writeASCII( filename + "-dy-ascii.txt", dy );
#endif // DEBUG_WRITE_DY_AS_ASCII
#endif // DEBUG_WRITE_DY_AS_PGM


#ifdef DEBUG_WRITE_MAG_AS_PGM
    cv::cuda::PtrStepSz32u mag( getHeight(),
                                getWidth(),
                                _h_debug_mag,
                                getWidth()*sizeof(uint32_t) );
    DebugImage::writePGMscaled( filename + "-mag.pgm", mag );
#ifdef DEBUG_WRITE_MAG_AS_ASCII
    DebugImage::writeASCII( filename + "-mag-ascii.txt", mag );
#endif // DEBUG_WRITE_MAG_AS_ASCII
#endif // DEBUG_WRITE_MAG_AS_PGM

#ifdef DEBUG_WRITE_MAP_AS_PGM
    cv::cuda::PtrStepSzb   map( getHeight(),
                                getWidth(),
                                _h_debug_map,
                                getWidth()*sizeof(uint8_t) );
    DebugImage::writePGMscaled( filename + "-map.pgm", map );
#ifdef DEBUG_WRITE_MAP_AS_ASCII
    DebugImage::writeASCII( filename + "-map-ascii.txt", map );
#endif // DEBUG_WRITE_MAP_AS_ASCII
#endif // DEBUG_WRITE_MAP_AS_PGM

#ifdef DEBUG_WRITE_HYSTEDGES_AS_PGM
    cv::cuda::PtrStepSzb   hystedges( getHeight(),
                                      getWidth(),
                                      _h_debug_hyst_edges,
                                      getWidth()*sizeof(uint8_t) );
    DebugImage::writePGMscaled( filename + "-hystedges.pgm", hystedges );
#endif // DEBUG_WRITE_HYSTEDGES_AS_PGM

    cv::cuda::PtrStepSzb   edges( getHeight(),
                                  getWidth(),
                                  _h_debug_edges,
                                  getWidth()*sizeof(uint8_t) );

#ifdef DEBUG_WRITE_EDGES_AS_PGM
    DebugImage::writePGMscaled( filename + "-edges.pgm", edges );
#endif // DEBUG_WRITE_EDGES_AS_PGM

#ifndef NDEBUG
#ifdef DEBUG_WRITE_EDGELIST_AS_PPM
    {
        /* Very basic debugging stage:
         * do we really put all of those edge points into the edge list
         * that are also present in the edge image?
         *
         * Confirmed that this works
         */
        vector<int2> out;
        _vote._all_edgecoords.debug_out( params._maxEdges, out );

        PtrStepSzbClone edgeclone( edges );
        DebugImage::plotPoints( out, edgeclone.e );
        DebugImage::writePPM( filename + "-edgelist.ppm", edgeclone.e );
#ifdef DEBUG_WRITE_EDGELIST_AS_ASCII
        DebugImage::writeASCII( filename + "-edgelist.txt", out );
#endif // DEBUG_WRITE_EDGELIST_AS_ASCII
    }
#endif // DEBUG_WRITE_EDGELIST_AS_PPM

#ifdef DEBUG_WRITE_VOTERS_AS_PPM
    {
        /* Debugging immediately after gradientDescent.
         * The list of TriplePoints has been created in
         * _vote._chained_edgecoords
         * These points have no before or after information yet.
         * The size of this list has not been copied to the host yet.
         */
        POP_CUDA_MEMCPY_TO_HOST_SYNC( &_vote._chained_edgecoords.host.size,
                                      _vote._chained_edgecoords.dev.size,
                                      sizeof(int) );

        vector<TriplePoint> out;
        _vote._chained_edgecoords.debug_out(  params._maxEdges, out );

        PtrStepSzbClone edgeclone( edges );
        DebugImage::plotPoints( out, edgeclone.e );
        DebugImage::writePPM( filename + "-voter-dots.ppm", edgeclone.e );
    }
#endif // DEBUG_WRITE_VOTERS_AS_PPM
#endif // NDEBUG

#ifndef NDEBUG
#ifdef DEBUG_WRITE_CHOSEN_AS_PPM
    {
        /* _chained_edgecoords.dev.size has been loaded into .host.size
         * _seed_indices has been created into this step.
         * _vote._seed_indices.dev.size, has been loaded into .host.soze
         * before returning.
         * The edge indices are all points that have received votes. No
         * filtering has taken place yet.
         * ... have the paths leading to these votes been stored?
         */
        if( _vote._chained_edgecoords.host.size > 0 && _vote._seed_indices.host.size > 0) {
            vector<TriplePoint> out;
            PtrStepSzbClone edgeclone( edges );
            _vote._chained_edgecoords.debug_out( params._maxEdges, out, EdgeListFilterCommittedOnly );
            DebugImage::plotPoints( out, edgeclone.e, true, DebugImage::GREEN );
#ifdef DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII
            DebugImage::writeASCII( filename + "-chosen-voter-chains.txt", out );
#endif // DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII

            out.clear();
            _vote._chained_edgecoords.debug_out( _vote._seed_indices, params._maxEdges, out );
            DebugImage::plotPoints( out, edgeclone.e, false, DebugImage::BLUE );
#ifdef DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII
            DebugImage::writeASCII( filename + "-chosen-dots.txt", out );
#endif // DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII

            DebugImage::writePPM( filename + "-chosen-dots.ppm", edgeclone.e );
        }
    }
#endif // DEBUG_WRITE_CHOSEN_AS_PPM
#endif // NDEBUG
#ifndef NDEBUG
#ifdef DEBUG_WRITE_LINKED_AS_PPM
    {
        if( _vote._chained_edgecoords.host.size > 0 && _vote._seed_indices.host.size > 0) {
            PtrStepSzbClone edgeclone( edges );
            DebugImage::BaseColor color = DebugImage::LAST;
#ifdef DEBUG_WRITE_LINKED_AS_ASCII
            ostringstream debug_ostr;
#endif // DEBUG_WRITE_LINKED_AS_ASCII
            bool do_print = false;
            for( int y=0; y<EDGE_LINKING_MAX_ARCS; y++ ) {
                vector<int2> out;
                for( int x=0; x<EDGE_LINKING_MAX_EDGE_LENGTH; x++ ) {
                    const int2& ref = _h_ring_output.ptr(y)[x];
                    if( ref.x != 0 || ref.y != 0 ) {
                        out.push_back( ref );
#ifdef DEBUG_WRITE_LINKED_AS_ASCII
                        debug_ostr << "(" << ref.x << "," << ref.y << ") ";
#endif // DEBUG_WRITE_LINKED_AS_ASCII
                    } else {
#ifdef DEBUG_WRITE_LINKED_AS_ASCII
                        debug_ostr << endl;
#endif // DEBUG_WRITE_LINKED_AS_ASCII
                        break;
                    }
                }
                if( out.size() != 0 ) {
                    DebugImage::plotPoints( out, edgeclone.e, false, color );
                    do_print = true;
                    color++;
                    if( color == DebugImage::WHITE ) color = DebugImage::LAST;
#ifdef DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                    PtrStepSzbClone e2( edges );
                    DebugImage::plotPoints( out, e2.e, true, DebugImage::BLUE );
                    ostringstream ostr;
                    ostr << filename << "-linked-dots-" << y << ".ppm";
                    cerr << "writing to " << ostr.str() << endl;
                    DebugImage::writePPM( ostr.str(), e2.e );
#endif // DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                }
            }
            if( do_print ) {
                DebugImage::writePPM( filename + "-linked-dots.ppm", edgeclone.e );
#ifdef DEBUG_WRITE_LINKED_AS_ASCII
                DebugImage::writeASCII( filename + "-linked-dots.txt", debug_ostr.str() );
#endif // DEBUG_WRITE_LINKED_AS_ASCII
            }
        }
    }
#endif // DEBUG_WRITE_LINKED_AS_PPM
#endif // NDEBUG
}

}; // namespace popart

