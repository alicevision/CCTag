/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "cuda/onoff.h"

// #include <string>
// #include <vector>
// #include <stdlib.h>
// #include <inttypes.h>
// #include <opencv2/core.hpp>
#include <cuda_runtime.h>

// #include <boost/thread/thread.hpp>
// #include <boost/thread/mutex.hpp>
// #include <boost/thread/condition.hpp>

#include "cuda/tag_threads.h"
#include "cuda/tag_cut.h"
#include "cuda/geom_ellipse.h"
#include "cuda/geom_matrix.h"

#include "cctag/Params.hpp"
// #include "cctag/Types.hpp"
// #include "cctag/ImageCut.hpp"
// #include "cctag/geometry/Ellipse.hpp"
// #include "cctag/geometry/Point.hpp"

namespace popart
{
class NearbyPoint;

struct ImageCenter
{
    bool                            _valid;
    const int                       _tagIndex;     // in
    const int                       _debug_numTags; // in - only for debugging
    const popart::geometry::ellipse _outerEllipse; // in
    popart::geometry::matrix3x3     _mT;
    popart::geometry::matrix3x3     _mInvT;
    const float                     _maxSemiAxis; // in
    const float2                    _center;       // in
    const int                       _vCutSize;     // in
    int                             _iterations;
    float                           _transformedEllipseMaxRadius;
    NearbyPoint*                    _cctag_pointer_buffer; // out

    ImageCenter( const int                       tagIndex,
                 const int                       debug_numTags,
                 const popart::geometry::ellipse outerEllipse,
                 const float2&                   center,
                 const int                       vCutSize,
                 NearbyPoint*                    cctag_pointer_buffer,
                 const cctag::Parameters& params )
        : _valid( true )
        , _tagIndex( tagIndex )
        , _debug_numTags( debug_numTags )
        , _outerEllipse( outerEllipse )
        , _maxSemiAxis( std::max( outerEllipse.a(), outerEllipse.b() ) )
        , _center( center )
        , _vCutSize( vCutSize )
        , _iterations( 0 )
        , _cctag_pointer_buffer( cctag_pointer_buffer )
    {
        const size_t gridNSample   = params._imagedCenterNGridSample;
        float        neighbourSize = params._imagedCenterNeighbourSize;

        if( _vCutSize < 2 ) {
            _valid = false;
            return;
        }

        if( _vCutSize != 22 ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                      << "    " << __func__ << " is called from CPU code with vCutSize "
                      << _vCutSize << " instead of 22" << std::endl;
            if( _vCutSize > 22 ) {
                exit( -1 );
            }
        }

        /* Determine the number of iterations by iteration */
        while( neighbourSize * _maxSemiAxis > 0.02 ) {
            _iterations += 1;
            neighbourSize /= (float)((gridNSample-1)/2) ;
        }

        _outerEllipse.makeConditionerFromEllipse( _mT );

        bool good = _mT.invert( _mInvT );
        if( not good ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                    << "    Conditioner matrix extracted from ellipse is not invertable" << std::endl
                    << "    Program logic error. Requires analysis before fixing." << std::endl
                    << std::endl;
            _valid = false;

            return;
        }

        popart::geometry::ellipse transformedEllipse;
        _outerEllipse.projectiveTransform( _mInvT, transformedEllipse );
        _transformedEllipseMaxRadius = std::max( transformedEllipse.a(), transformedEllipse.b() );
    }

    void setInvalid( )
    {
        _valid = false;
    }
};

}; // namespace popart

