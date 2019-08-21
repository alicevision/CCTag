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

/*************************************************************
 * functions writing PGM (1-layer uncompressed grayscale maps)
 *************************************************************/

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
            T f = std::max<T>( 0, std::min<T>( 255, plane.at(x,y) ) );
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

/*************************************************************
 * functions writing PPM (3-layered uncompressed pixel maps)
 *************************************************************/

static void writePlaneP6( const std::string& filename, const uint8_t* data, int cols, int rows )
{
    std::ofstream of( filename.c_str() );
    of << "P6" << std::endl
       << cols << " " << rows << std::endl
       << "255" << std::endl;

    of.write( (char*)data, 3 * cols * rows );
}

static void writePlanePPM_scaled( const std::string& filename, const Plane<Color>& plane )
{
    std::cerr << "Writing scaled ppm file " << filename << std::endl;

    float fminval = std::numeric_limits<float>::max();
    float fmaxval = std::numeric_limits<float>::min();
    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            const Color& f = plane.at(x,y);
            fminval = std::min<float>( std::min<float>( fminval, f.r() ),
                                       std::min<float>( f.g(),   f.b() ) );
            fmaxval = std::max<float>( std::max<float>( fmaxval, f.r() ),
                                       std::max<float>( f.g(),   f.b() ) );
        }
    }
    fmaxval = 255.0 / ( fmaxval - fminval );

    uint8_t data[ 3 * plane.getCols() * plane.getRows() ];
    Plane<uint8_t> dst( data, plane.getRows(), 3 * plane.getCols() );

    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            const Color& f = plane.at(x,y);
            float r = ( (float)f.r() - fminval ) * fmaxval;
            float g = ( (float)f.g() - fminval ) * fmaxval;
            float b = ( (float)f.b() - fminval ) * fmaxval;
            dst.at(3*x+0,y) = (uint8_t)r;
            dst.at(3*x+1,y) = (uint8_t)g;
            dst.at(3*x+2,y) = (uint8_t)b;
        }
    }

    writePlaneP6( filename, data, dst.getCols(), dst.getRows() );
}

static void writePlanePPM_unscaled( const std::string& filename, const Plane<Color>& plane )
{
    std::cerr << "Writing unscaled ppm file " << filename << std::endl;

    uint8_t data[ 3 * plane.getCols() * plane.getRows() ];
    Plane<uint8_t> dst( data, plane.getRows(), 3 * plane.getCols() );

    for( size_t y=0; y<plane.getRows(); y++ ) { 
        for( size_t x=0; x<plane.getCols(); x++ ) {
            const Color& val = plane.at(x,y);
            uint8_t r = std::max( (uint8_t)0, std::min( (uint8_t)255, (uint8_t)val.r() ) );
            uint8_t g = std::max( (uint8_t)0, std::min( (uint8_t)255, (uint8_t)val.g() ) );
            uint8_t b = std::max( (uint8_t)0, std::min( (uint8_t)255, (uint8_t)val.b() ) );
            dst.at(3*x+0,y) = r;
            dst.at(3*x+1,y) = g;
            dst.at(3*x+2,y) = b;
        }
    }

    writePlaneP6( filename, data, dst.getCols(), dst.getRows() );
}

void writePlanePPM( const std::string&     filename,
                    const Plane<Color>&    plane,
                    bool                   scaled )
{
    if( scaled )
        writePlanePPM_scaled( filename, plane );
    else
        writePlanePPM_unscaled( filename, plane );
}

}; // namespace cctag

