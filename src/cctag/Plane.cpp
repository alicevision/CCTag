/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <string>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>

#include "cctag/Plane.hpp"

namespace cctag
{

static void writePlaneP5( const std::string& filename, const uint8_t* data, int cols, int rows )
{
    std::ofstream of( filename.c_str() );
    of << "P5" << std::endl
       << cols << " " << rows << std::endl
       << "255" << std::endl;

    of.write( (char*)data, cols * rows );
}

template<typename T>
static void writePlanePGM_scaled( const std::string& filename, const Plane<T>& plane )
{
    std::cerr << "Writing scaled pgm file " << filename << std::endl;

    T minval = std::numeric_limits<T>::max();
    T maxval = std::numeric_limits<T>::min();
    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            T f = plane.at(x,y);
            minval = std::min<T>( minval, f );
            maxval = std::max<T>( maxval, f );
        }
    }
    float fminval = (float)minval;
    float fmaxval = 255.0 / ( (float)maxval - fminval );

    uint8_t data[ plane.getCols() * plane.getRows() ];
    Plane<uint8_t> dst( data, plane.getRows(), plane.getCols() );

    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            T f = plane.at(x,y);
            float outf = ( (float)f - fminval ) * fmaxval;
            dst.at(x,y) = (uint8_t)outf;
        }
    }

    writePlaneP5( filename, data, dst.getCols(), dst.getRows() );
}

template<typename T>
static void writePlanePGM_unscaled( const std::string& filename, const Plane<T>& plane )
{
    std::cerr << "Writing unscaled pgm file " << filename << std::endl;

    uint8_t data[ plane.getCols() * plane.getRows() ];
    Plane<uint8_t> dst( data, plane.getRows(), plane.getCols() );

    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            T f = std::min<T>( 0, std::max<T>( 255, plane.at(x,y) ) );
            dst.at(x,y) = (uint8_t)f;
        }
    }

    writePlaneP5( filename, data, dst.getCols(), dst.getRows() );
}

void writePlanePGM( const std::string& filename, const Plane<uint8_t>& plane, bool scaled )
{
    if( scaled )
        writePlanePGM_scaled<uint8_t>( filename, plane );
    else
        writePlanePGM_unscaled<uint8_t>( filename, plane );
}

void writePlanePGM( const std::string& filename, const Plane<int8_t>& plane, bool scaled )
{
    if( scaled )
        writePlanePGM_scaled<int8_t>( filename, plane );
    else
        writePlanePGM_unscaled<int8_t>( filename, plane );
}

void writePlanePGM( const std::string& filename, const Plane<uint16_t>& plane, bool scaled )
{
    if( scaled )
        writePlanePGM_scaled<uint16_t>( filename, plane );
    else
        writePlanePGM_unscaled<uint16_t>( filename, plane );
}

void writePlanePGM( const std::string& filename, const Plane<int16_t>& plane, bool scaled )
{
    if( scaled )
        writePlanePGM_scaled<int16_t>( filename, plane );
    else
        writePlanePGM_unscaled<int16_t>( filename, plane );
}

}; // namespace cctag

