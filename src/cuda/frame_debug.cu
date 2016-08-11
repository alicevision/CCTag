/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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

// #include "../cctag/cmdline.hpp"

#include "frame.h"
#include "debug_image.h"
#include "edge_list.h"
#include "cmp_list.h"
#include "assist.h"

#undef CHATTY_WRITE_DEBUG_PLANE

namespace popart {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

void Frame::hostDebugDownload( const cctag::Parameters& params )
{
    delete [] _h_debug_hyst_edges;

    _h_debug_hyst_edges = new unsigned char[ getWidth() * getHeight() ];

    POP_SYNC_CHK;

    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_hyst_edges, getWidth() * sizeof(uint8_t),
                              _d_hyst_edges.data, _d_hyst_edges.step,
                              _d_hyst_edges.cols * sizeof(uint8_t),
                              _d_hyst_edges.rows,
                              cudaMemcpyDeviceToHost, _stream );

    POP_CUDA_MEMCPY_2D_ASYNC( _h_edges.data, _h_edges.step,
                              _d_edges.data, _d_edges.step,
                              _d_edges.cols * sizeof(uint8_t),
                              _d_edges.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CHK_CALL_IFSYNC;
}

void Frame::hostDebugCompare( unsigned char* pix )
{
#ifndef NDEBUG
#ifdef DEBUG_WRITE_ORIGINAL_AS_PGM
    bool found_mistake = false;
    size_t mistake_ct = 0;

    for( int h=0; h<_d_plane.rows; h++ ) {
        for( int w=0; w<_d_plane.cols; w++ ) {
            size_t pos = h*_d_plane.cols+w;
            if( pix[pos] != _h_plane.data[pos] ) {
                mistake_ct++;
                if( found_mistake == false ) {
                    found_mistake = true;
                    cerr << "Found first error at (" << w << "," << h << "): "
                         << "orig " << pix[pos]
                         << "copy " << _h_plane.data[pos]
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
#endif // DEBUG_WRITE_ORIGINAL_AS_PGM
#endif // NDEBUG
}

void Frame::writeHostDebugPlane( string filename, const cctag::Parameters& params )
{
    filename = params._debugDir + filename;

    string s;

#ifdef DEBUG_WRITE_ORIGINAL_AS_PGM
    const cv::cuda::PtrStepSzb& b = _h_plane;
    DebugImage::writePGM( filename + "-01.pgm", b );
#ifdef DEBUG_WRITE_ORIGINAL_AS_ASCII
    DebugImage::writeASCII( filename + "-01-img-ascii.txt", b );
#endif // DEBUG_WRITE_ORIGINAL_AS_ASCII
#endif // WRITE_ORIGINAL_AS_PGM

#ifdef DEBUG_WRITE_DX_AS_PGM
    cv::cuda::PtrStepSz16s dx( getHeight(),
                               getWidth(),
                               _h_dx,
                               getWidth()*sizeof(int16_t) );
    DebugImage::writePGMscaled( filename + "-02-dx.pgm", dx );
#ifdef DEBUG_WRITE_DX_AS_ASCII
    DebugImage::writeASCII( filename + "-02-dx-ascii.txt", dx );
#endif // DEBUG_WRITE_DX_AS_ASCII
#endif // DEBUG_WRITE_DX_AS_PGM

#ifdef DEBUG_WRITE_DY_AS_PGM
    cv::cuda::PtrStepSz16s dy( getHeight(),
                               getWidth(),
                               _h_dy,
                               getWidth()*sizeof(int16_t) );
    DebugImage::writePGMscaled( filename + "-02-dy.pgm", dy );
#ifdef DEBUG_WRITE_DY_AS_ASCII
    DebugImage::writeASCII( filename + "-02-dy-ascii.txt", dy );
#endif // DEBUG_WRITE_DY_AS_ASCII
#endif // DEBUG_WRITE_DY_AS_PGM


#ifdef DEBUG_WRITE_MAG_AS_PGM
    const cv::cuda::PtrStepSz32u& mag = _h_mag;
    DebugImage::writePGMscaled( filename + "-03-mag.pgm", mag );
#ifdef DEBUG_WRITE_MAG_AS_ASCII
    DebugImage::writeASCII( filename + "-03-mag-ascii.txt", mag );
#endif // DEBUG_WRITE_MAG_AS_ASCII
#endif // DEBUG_WRITE_MAG_AS_PGM

#ifdef DEBUG_WRITE_MAP_AS_PGM
    cv::cuda::PtrStepSzb   map( getHeight(),
                                getWidth(),
                                _h_debug_map,
                                getWidth()*sizeof(uint8_t) );
    DebugImage::writePGMscaled( filename + "-03-map.pgm", map );
#ifdef DEBUG_WRITE_MAP_AS_ASCII
    DebugImage::writeASCII( filename + "-03-map-ascii.txt", map );
#endif // DEBUG_WRITE_MAP_AS_ASCII
#endif // DEBUG_WRITE_MAP_AS_PGM

#ifdef DEBUG_WRITE_HYSTEDGES_AS_PGM
    cv::cuda::PtrStepSzb   hystedges( getHeight(),
                                      getWidth(),
                                      _h_debug_hyst_edges,
                                      getWidth()*sizeof(uint8_t) );
    DebugImage::writePGMscaled( filename + "-04-hystedges.pgm", hystedges );
#endif // DEBUG_WRITE_HYSTEDGES_AS_PGM

#ifndef NDEBUG
    const cv::cuda::PtrStepSzb&  edges = _h_edges;
#endif // NDEBUG

#ifdef DEBUG_WRITE_EDGES_AS_PGM
    DebugImage::writePGMscaled( filename + "-05-edges.pgm", edges );
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
        _vote._all_edgecoords.debug_out( EDGE_POINT_MAX, out );

        PtrStepSzbNull edgelistplane( edges.cols, edges.rows );
        DebugImage::plotPoints( out, edgelistplane.e, false, DebugImage::BLUE );
        DebugImage::writePGMscaled( filename + "-05-edgelist.pgm", edgelistplane.e );
#ifdef DEBUG_WRITE_EDGELIST_AS_ASCII
        int2cmp c;
        std::sort( out.begin(), out.end(), c );
        DebugImage::writeASCII( filename + "-05-edgelist.txt", out );
#endif // DEBUG_WRITE_EDGELIST_AS_ASCII
    }
#endif // DEBUG_WRITE_EDGELIST_AS_PPM

#ifdef DEBUG_WRITE_VOTERS_AS_PPM
    {
        /* Debugging immediately after gradientDescent.
         * The list of TriplePoints has been created in
         * _voters
         * These points have no before or after information yet.
         * The size of this list has not been copied to the host yet.
         */
        POP_CUDA_MEMCPY_TO_HOST_SYNC( &_voters.host.size,
                                      _voters.dev.getSizePtr(),
                                      sizeof(int) );

        vector<TriplePoint> out;
        _voters.debug_out(  EDGE_POINT_MAX, out );

        PtrStepSzbNull edgelistplane( edges.cols, edges.rows );
        DebugImage::plotPoints( out, edgelistplane.e, false, DebugImage::BLUE );
        DebugImage::writePGMscaled( filename + "-06-voter-dots.pgm", edgelistplane.e );
#ifdef DEBUG_WRITE_VOTERS_AS_ASCII
        tp_cmp c;
        std::sort( out.begin(), out.end(), c );
        DebugImage::writeASCII( filename + "-06-voters.txt", out );
#endif // DEBUG_WRITE_VOTERS_AS_ASCII
    }
#endif // DEBUG_WRITE_VOTERS_AS_PPM
#endif // NDEBUG

#ifndef NDEBUG
#ifdef DEBUG_WRITE_CHOSEN_AS_PPM
    {
        /* _voters.dev.size has been loaded into .host.size
         * _inner_points has been created into this step.
         * _inner_points.dev.size, has been loaded into .host.soze
         * before returning.
         * The edge indices are all points that have received votes. No
         * filtering has taken place yet.
         * ... have the paths leading to these votes been stored?
         */
        if( _voters.host.size > 0 && _inner_points.host.size > 0) {
            vector<TriplePoint> out;
            PtrStepSzbClone edgeclone( edges );
            _voters.debug_out( EDGE_POINT_MAX, out, EdgeListFilterCommittedOnly );
            DebugImage::plotPoints( out, edgeclone.e, true, DebugImage::GREEN );
#ifdef DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII
            DebugImage::writeASCII( filename + "-07-chosen-voter-chains.txt", out );
#endif // DEBUG_WRITE_CHOSEN_VOTERS_AS_ASCII

            out.clear();
            _voters.debug_out( _inner_points, EDGE_POINT_MAX, out );
            DebugImage::plotPoints( out, edgeclone.e, false, DebugImage::BLUE );
#ifdef DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII
            DebugImage::writeASCII( filename + "-07-chosen-dots.txt", out );
#endif // DEBUG_WRITE_CHOSEN_ELECTED_AS_ASCII

            DebugImage::writePPM( filename + "-07-chosen-dots.ppm", edgeclone.e );

            PtrStepSzbNull edgelistplane( edges.cols, edges.rows );
            DebugImage::plotPoints( out, edgelistplane.e, false, DebugImage::BLUE );
            DebugImage::writePGMscaled( filename + "-07-chosen-dots.pgm", edgelistplane.e );
        }
    }
#endif // DEBUG_WRITE_CHOSEN_AS_PPM
#endif // NDEBUG
#ifndef EDGE_LINKING_HOST_SIDE
#ifndef NDEBUG
#ifdef DEBUG_WRITE_LINKED_AS_PPM
    {
        cerr << "Enter link writing block" << endl;
#ifdef DEBUG_WRITE_LINKED_AS_ASCII
        const bool write_linked_as_ascii = true;
#else // DEBUG_WRITE_LINKED_AS_ASCII
        const bool write_linked_as_ascii = false;
#endif // DEBUG_WRITE_LINKED_AS_ASCII
        if( _voters.host.size > 0 && _inner_points.host.size > 0) {
            PtrStepSzbClone edgeclone( edges );
            ostringstream debug_ostr;
            bool do_print = false;
#ifdef DEBUG_LINKED_USE_INT4_BUFFER
            for( int y=0; y<EDGE_LINKING_MAX_ARCS; y++ ) {
                bool write_linked_as_ascii_first_entry_in_line_written = false;
                vector<int2> out_blue;
                vector<int2> out_green;
                for( int x=0; x<EDGE_LINKING_MAX_EDGE_LENGTH; x++ ) {
                    const int4& ref = _h_ring_output.ptr(y)[x];
                    if( ref.x != 0 || ref.y != 0 ) {
                        int2 dat = make_int2( ref.x, ref.y );
                        if( ref.z == 0 )
                            out_blue.push_back( dat );
                        else
                            out_green.push_back( dat );
                        if( write_linked_as_ascii ) {
                            if( not write_linked_as_ascii_first_entry_in_line_written ) {
                                write_linked_as_ascii_first_entry_in_line_written = true;
                                debug_ostr << "Arc " << y << ": ";
                            }
                            debug_ostr << "(" << ref.x << "," << ref.y << ":";
                            if( ref.z<100 )
                                debug_ostr << "L" << ref.z;
                            else
                                debug_ostr << "R" << (ref.z-100);
                            debug_ostr << ":" << ref.w << ") ";
                        }
                    } else {
                        if( write_linked_as_ascii && write_linked_as_ascii_first_entry_in_line_written ) {
                            debug_ostr << endl;
                        }
                        break;
                    }
                }
                if( out_green.size() != 0 || out_blue.size() != 0 ) {
                    do_print = true;
                    PtrStepSzbClone e2( edges );
                    DebugImage::plotPoints( out_blue,  e2.e, true, DebugImage::BLUE );
                    DebugImage::plotPoints( out_green, e2.e, false, DebugImage::GREEN );
                    ostringstream ostr;
                    ostr << filename << "-linked-dots-" << y << ".ppm";
                    // cerr << "writing to " << ostr.str() << endl;
                    DebugImage::writePPM( ostr.str(), e2.e );
                }
            }
#else // DEBUG_LINKED_USE_INT4_BUFFER
            DebugImage::BaseColor color = DebugImage::LAST;
            for( int y=0; y<EDGE_LINKING_MAX_ARCS; y++ ) {
                if( write_linked_as_ascii )
                    debug_ostr << "Arc " << y << ": ";
                vector<int2> out;
                for( int x=0; x<EDGE_LINKING_MAX_EDGE_LENGTH; x++ ) {
                    const int2& ref = _h_ring_output.ptr(y)[x];
                    if( ref.x != 0 || ref.y != 0 ) {
                        out.push_back( ref );
                        if( write_linked_as_ascii ) {
                            debug_ostr << "(" << ref.x << "," << ref.y << ") ";
                        }
                    } else {
                        if( write_linked_as_ascii ) debug_ostr << endl;
                        break;
                    }
                }
                if( out.size() != 0 ) {
                    do_print = true;
#ifdef DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                    PtrStepSzbClone e2( edges );
                    DebugImage::plotPoints( out, e2.e, true, DebugImage::BLUE );
                    ostringstream ostr;
                    ostr << filename << "-linked-dots-" << y << ".ppm";
                    // cerr << "writing to " << ostr.str() << endl;
                    DebugImage::writePPM( ostr.str(), e2.e );
#else // DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                    DebugImage::plotPoints( out, edgeclone.e, false, color );
                    color++;
                    if( color == DebugImage::WHITE ) color = DebugImage::LAST;
#endif // DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                } else {
                    cerr << "Not plotting in _h_ring_output, out.size==0" << endl;
                }
            }
#endif // DEBUG_LINKED_USE_INT4_BUFFER
            if( do_print ) {
#ifndef DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                DebugImage::writePPM( filename + "-linked-dots.ppm", edgeclone.e );
#endif // not DEBUG_WRITE_LINKED_AS_PPM_INTENSE
                if( write_linked_as_ascii ) {
                    DebugImage::writeASCII( filename + "-linked-dots.txt", debug_ostr.str() );
                }
            }
        } else {
            cerr << "Not plotting anything from _h_ring_output." << endl
                 << "    # chained edge coords: " << _voters.host.size << endl
                 << "    # seed indices: " << _inner_points.host.size << endl
                 << "    # _h_ring_output dimensions: (" << _h_ring_output.cols << "," << _h_ring_output.rows << ")" << endl;
        }
        cerr << "Leave link writing block" << endl;
    }
#endif // DEBUG_WRITE_LINKED_AS_PPM
#endif // NDEBUG
#endif // EDGE_LINKING_HOST_SIDE
}

}; // namespace popart

