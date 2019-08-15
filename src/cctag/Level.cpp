/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <sstream>

#include <cctag/Level.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/filter/thinning.hpp>
#include "cctag/utils/Talk.hpp"
#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif
#include "cctag/PlaneCV.hpp"
#include "cctag/nocuda/resize.hpp"
#include "cctag/nocuda/recode.hpp"

namespace cctag {

Level::Level( std::size_t width, std::size_t height, int debug_info_level, bool cuda_allocates )
    : _level( debug_info_level )
    , _cuda_allocates( cuda_allocates )
    , _mat_initialized_from_cuda( false )
    , _cols( width )
    , _rows( height )
    , _temp( height, width )
#ifdef CCTAG_EXTRA_LAYER_DEBUG
    , _edgesNotThin( height, width )
#endif
{
    if( _cuda_allocates ) {
        _src   = nullptr;
        _dx    = nullptr;
        _dy    = nullptr;
        _mag   = nullptr;
        _edges = nullptr;
    } else {
        // Allocation
        _src   = new Plane<uint8_t>( height, width );
        _dx    = new Plane<int16_t>( height, width );
        _dy    = new Plane<int16_t>( height, width );
        _mag   = new Plane<int16_t>( height, width );
        _edges = new Plane<uint8_t>( height, width );
    }
}

Level::~Level( )
{
    delete _src;
    delete _dx;
    delete _dy;
    delete _mag;
    delete _edges;
}

void Level::setLevel( const Plane<uint8_t>& src,
                      float thrLowCanny,
                      float thrHighCanny,
                      const cctag::Parameters* params )
{
    if( _cuda_allocates ) {
        std::cerr << "This function makes no sense with CUDA in " << __FUNCTION__ << ":" << __LINE__ << std::endl;
        exit( -__LINE__ );
    }

#if 0
#if 0
    cv::resize( planeToMat( src ), planeToMat( *_src ), cv::Size( _src->getCols(),_src->getRows() ) );
    // ASSERT TODO : check that the data are allocated here
    // Compute derivative and canny edge extraction.
    cvRecodedCanny( *_src, *_edges, *_dx, *_dy,
                    thrLowCanny * 256, thrHighCanny * 256,
                    3 | CV_CANNY_L2_GRADIENT,
                    _level, params );
#else
    cctag::resize( src, *_src );
    cvRecodedCanny( *_src, *_edges, *_dx, *_dy,
                    thrLowCanny * 256, thrHighCanny * 256,
                    3 | CV_CANNY_L2_GRADIENT,
                    _level, params );
#endif
#else
    cctag::resize( src, *_src );
    cctag::recodedCanny( *_src, *_edges, *_dx, *_dy,
                         thrLowCanny * 256, thrHighCanny * 256,
                         _level,
                         params );

#if 1
    Plane<uint8_t> testEdges( _edges->getRows(), _edges->getCols() );
    Plane<int16_t> diffEdges( _edges->getRows(), _edges->getCols() );
    cvRecodedCanny( *_src, testEdges, *_dx, *_dy,
                    thrLowCanny * 256, thrHighCanny * 256,
                    3 | CV_CANNY_L2_GRADIENT,
                    _level, params );
    for( int y=0; y<diffEdges.getRows(); y++ )
        for( int x=0; x<diffEdges.getCols(); x++ )
        {
            diffEdges.at(x,y) = (int16_t)_edges->at(x,y) - (int16_t)testEdges.at(x,y);
        }
    std::ostringstream o1, o2, o3;
    o1 << "canny-" << _level << "-cv.pgm";
    o2 << "canny-" << _level << "-nocv.pgm";
    o3 << "canny-" << _level << "-diff-cv-nocv.pgm";
    writePlanePGM( o1.str(), testEdges, SCALED_WRITING );
    writePlanePGM( o2.str(), *_edges, SCALED_WRITING );
    writePlanePGM( o3.str(), diffEdges, SCALED_WRITING );
#endif

#endif
    // Perform the thinning.

#ifdef CCTAG_EXTRA_LAYER_DEBUG
    _edgesNotThin = _edges->clone();
#endif
  
    thin( *_edges, _temp );
}

#ifdef CCTAG_WITH_CUDA
void Level::setLevel( cctag::TagPipe*         cuda_pipe,
                      const cctag::Parameters& params )
{
    if( ! _cuda_allocates ) {
        std::cerr << "This function makes no sense without CUDA in " << __FUNCTION__ << ":" << __LINE__ << std::endl;
        exit( -__LINE__ );
    }

    _src   = cuda_pipe->getPlane( _level );
    _dx    = cuda_pipe->getDx( _level );
    _dy    = cuda_pipe->getDy( _level );
    _mag   = cuda_pipe->getMag( _level );
    _edges = cuda_pipe->getEdges( _level );
}
#endif // CCTAG_WITH_CUDA

Plane<uint8_t>& Level::getSrc()
{
    return *_src;
}

#ifdef CCTAG_EXTRA_LAYER_DEBUG
const Plane<uint8_t>& Level::getCannyNotThin() const
{
    return _edgesNotThin;
}
#endif

Plane<int16_t>& Level::getDx() const
{
    return *_dx;
}

Plane<int16_t>& Level::getDy() const
{
    return *_dy;
}

Plane<int16_t>& Level::getMag() const
{
    return *_mag;
}

Plane<uint8_t>& Level::getEdges() const
{
    return *_edges;
}

}
