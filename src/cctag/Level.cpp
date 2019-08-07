/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Level.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/filter/thinning.hpp>
#include "cctag/utils/Talk.hpp"
#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif
#include <opencv2/imgproc/types_c.h>

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

    cv::resize( src.getMat(), _src->getMat(), cv::Size( _src->getCols(),_src->getRows() ) );
    // ASSERT TODO : check that the data are allocated here
    // Compute derivative and canny edge extraction.
    cv::Mat edges = _edges->getMat();
    cv::Mat dx    = _dx->getMat();
    cv::Mat dy    = _dy->getMat();
    cv::Mat temp  = _temp.getMat();
    cvRecodedCanny( _src->getMat(), edges, dx, dy,
                    thrLowCanny * 256, thrHighCanny * 256,
                    3 | CV_CANNY_L2_GRADIENT,
                    _level, params );
    // Perform the thinning.

#ifdef CCTAG_EXTRA_LAYER_DEBUG
    _edgesNotThin = _edges->clone();
#endif
  
    thin(edges,temp);
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
