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
#include <cctag/cuda/cctag_cuda_runtime.h>
#include <map>
#include "debug_macros.hpp"

#include "frame.h"
#include "debug_image.h"
#include "edge_list.h"
#include "cmp_list.h"
#include "assist.h"

#undef CHATTY_WRITE_DEBUG_PLANE

namespace cctag {

using namespace std;

/*************************************************************
 * Frame
 *************************************************************/

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
}

}; // namespace cctag

