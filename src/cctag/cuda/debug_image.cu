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
//#include <sys/stat.h>
#include <map>
// #include "debug_macros.hpp"

#include "frame.h"
#include "debug_image.h"
#include "assist.h"

namespace cctag {

using namespace std;

/*************************************************************
 * DebugeImage
 *************************************************************/

DebugImage::RandomColorMap DebugImage::randomColorMap;

__host__
void DebugImage::writePGM( const string& filename, const cv::cuda::PtrStepSzb& plane )
{
    assert( plane.data );

    ofstream of( filename.c_str() );
    of << "P5" << endl
       << plane.cols << " " << plane.rows << endl
       << "255" << endl;
    of.write( (char*)plane.data, plane.cols * plane.rows );

    uint32_t ct = 0;
    for( int x=0; x<plane.cols; x++ ) {
        for( int y=0; y<plane.rows; y++ ) {
            if( plane.ptr(y)[x] != 0 ) ct++;
        }
    }
    cerr << "Writing pgm file " << filename << ": "
         << ct << " non-null pixels" << endl;
}

template<class T>
__host__
void DebugImage::writePGMscaled_T( const string&                 filename,
                                   const cv::cuda::PtrStepSz<T>& plane )
{
    uint32_t ct = 0;
    for( int x=0; x<plane.cols; x++ ) {
        for( int y=0; y<plane.rows; y++ ) {
            if( plane.ptr(y)[x] != 0 ) ct++;
        }
    }
    cerr << "Writing scaled pgm file " << filename << ": "
         << ct << " non-null pixels" << endl;

    T minval = std::numeric_limits<T>::max();
    T maxval = std::numeric_limits<T>::min();
    // for( uint32_t i=0; i<plane.rows*plane.cols; i++ )
    for( size_t i=0; i<plane.rows; i++ ) {
        for( size_t j=0; j<plane.cols; j++ ) {
            T f = plane.ptr(i)[j];
            // float f = plane.data[i];
            minval = min( minval, f );
            maxval = max( maxval, f );
        }
    }

    ofstream of( filename.c_str() );
    of << "P5" << endl
       << plane.cols << " " << plane.rows << endl
       << "255" << endl;

    float fmaxval = 255.0 / ( (float)maxval - (float)minval );
    for( uint32_t i=0; i<plane.rows*plane.cols; i++ ) {
        T f = plane.data[i];
        float outf = ( (float)f - (float)minval ) * fmaxval;
        unsigned char uc = (unsigned char)outf;
        of << uc;
    }
}
__host__
void DebugImage::writePGMscaled( const std::string& filename,
                                 const cv::cuda::PtrStepSz<float>& plane )
{
    writePGMscaled_T( filename, plane );
}
__host__
void DebugImage::writePGMscaled( const std::string& filename,
                                 const cv::cuda::PtrStepSz<uint8_t>& plane )
{
    writePGMscaled_T( filename, plane );
}
__host__
void DebugImage::writePGMscaled( const std::string& filename,
                                 const cv::cuda::PtrStepSz<int16_t>& plane )
{
    writePGMscaled_T( filename, plane );
}
__host__
void DebugImage::writePGMscaled( const std::string& filename,
                                 const cv::cuda::PtrStepSz<uint32_t>& plane )
{
    writePGMscaled_T( filename, plane );
}

__host__
void DebugImage::writePPM( const string& filename, const cv::cuda::PtrStepSzb& plane )
{
    ofstream of( filename.c_str() );
    of << "P6" << endl
       << plane.cols << " " << plane.rows << endl
       << "255" << endl;

    for( uint32_t i=0; i<plane.rows*plane.cols; i++ ) {
        unsigned char f = plane.data[i];
        const DebugImage::RandomColor& c = DebugImage::randomColorMap.get( f );
        of << c.r << c.g << c.b;
    }
}

template<class T>
__host__
void DebugImage::writeASCII_T( const string& filename, const cv::cuda::PtrStepSz<T>& plane, int width )
{
    ofstream of( filename.c_str() );
    // for( int y=0; y<getHeight(); y++ ) for( int x=0; x<getWidth(); x++ )
    for( int y=0; y<plane.rows; y++ ) {
        for( int x=0; x<plane.cols; x++ )
        {
            int val = plane.ptr(y)[x];
            if( width != 0 )
                of << setw(width) << val << " ";
            else
                of << val << " ";
        }
        of << endl;
    }
}

__host__
void DebugImage::writeASCII( const string& filename,
                             const cv::cuda::PtrStepSz<float>& plane )
{
    writeASCII_T( filename, plane );
}
__host__
void DebugImage::writeASCII( const string& filename,
                             const cv::cuda::PtrStepSz<uint8_t>& plane )
{
    writeASCII_T( filename, plane, 3 );
}
__host__
void DebugImage::writeASCII( const string& filename,
                             const cv::cuda::PtrStepSz<int16_t>& plane )
{
    writeASCII_T( filename, plane, 3 );
}
__host__
void DebugImage::writeASCII( const string& filename,
                             const cv::cuda::PtrStepSz<uint32_t>& plane )
{
    writeASCII_T( filename, plane );
}
__host__
void DebugImage::writeASCII( const string&           filename,
                             const std::vector<int>& list )
{
    ofstream of( filename.c_str() );

    vector<int>::const_iterator it  = list.begin();
    vector<int>::const_iterator end = list.end();
    for( ; it!=end; it++ ) {
        of << *it << endl;
    }
}
__host__
void DebugImage::writeASCII( const string&            filename,
                             const std::vector<int2>& list )
{
    ofstream of( filename.c_str() );

    vector<int2>::const_iterator it  = list.begin();
    vector<int2>::const_iterator end = list.end();
    for( ; it!=end; it++ ) {
        of << it->x << " " << it->y << endl;
    }
}

#ifndef NDEBUG
__host__
void DebugImage::writeASCII( const string&                   filename,
                             const std::vector<TriplePoint>& list )
{
    ofstream of( filename.c_str() );

    vector<TriplePoint>::const_iterator it  = list.begin();
    vector<TriplePoint>::const_iterator end = list.end();
    for( ; it!=end; it++ ) {
        it->debug_out( of );
        of << endl;
    }
}

__host__
void DebugImage::writeASCII( const string& filename,
                             const string& data )
{
    ofstream of( filename.c_str() );

    of << data;
}
#endif // NDEBUG

void DebugImage::normalizeImage( cv::cuda::PtrStepSzb img, bool normalize )
{
    if( not normalize ) return;

    /* All images points that are non-null are normalized to 1.
     */
    for( uint32_t x=0; x<img.cols; x++ ) {
        for( uint32_t y=0; y<img.rows; y++ ) {
            if( img.ptr(y)[x] != BLACK ) img.ptr(y)[x] = GREY1;
        }
    }
}

int DebugImage::getColor( BaseColor b )
{
    if( b < LAST ) return b;

    return ( LAST + random() % ( 255 - LAST ) );
}

void DebugImage::plotPoints( const vector<TriplePoint>& v, cv::cuda::PtrStepSzb img, bool normalize, BaseColor b )
{
    normalizeImage( img, normalize );

    vector<TriplePoint>::const_iterator cit, cend;
    cend = v.end();
    cout << "Plotting in image of size " << img.cols << " x " << img.rows << endl;
    for( cit=v.begin(); cit!=cend; cit++ ) {
        if( outOfBounds( cit->coord.x, cit->coord.y, img ) ) {
            cout << "Coord of point (" << cit->coord.x << "," << cit->coord.y << ") is out of bounds (line " << __LINE__ << ")" << endl;
        } else {
            img.ptr(cit->coord.y)[cit->coord.x] = getColor( b );
        }
    }
}

void DebugImage::plotPoints( const vector<int2>& v, cv::cuda::PtrStepSzb img, bool normalize, BaseColor b )
{
    normalizeImage( img, normalize );

    vector<int2>::const_iterator cit, cend;
    cend = v.end();
    cout << "Plotting " << v.size() << " int2 coordinates into image of size " << img.cols << " x " << img.rows << endl;
    for( cit=v.begin(); cit!=cend; cit++ ) {
        if( outOfBounds( cit->x, cit->y, img ) ) {
            cout << "Coord of point (" << cit->x << "," << cit->y << ") is out of bounds (line " << __LINE__ << ")" << endl;
        } else {
            img.ptr(cit->y)[cit->x] = getColor( b );
        }
    }
}

void DebugImage::plotOneLine( int2 from, int2 to, cv::cuda::PtrStepSzb img, int color )
{
    int  absx = abs(from.x-to.x);
    int  absy = abs(from.y-to.y);
    if( absx >= absy ) {
        if( from.x > to.x ) {
            plotOneLine( to, from, img, color );
        } else {
            for( int xko=from.x; xko<=to.x; xko++ ) {
                int yko = from.y + (int)roundf( ( to.y - from.y ) * (float)(xko-from.x) / (float)absx ); 
                if( not outOfBounds( xko, yko, img ) ) {
                    img.ptr(yko)[xko] = color;
                }
            }
        }
    } else {
        if( from.y > to.y ) {
            plotOneLine( to, from, img, color );
        } else {
            for( int yko=from.y; yko<=to.y; yko++ ) {
                int xko = from.x + (int)roundf( ( to.x - from.x ) * (float)(yko-from.y) / (float)absy ); 
                if( not outOfBounds( xko, yko, img ) ) {
                    img.ptr(yko)[xko] = color;
                }
            }
        }
    }
}

#ifndef NDEBUG
void DebugImage::plotLines( EdgeList<TriplePoint>& points,
                            int                    maxSize,
                            cv::cuda::PtrStepSzb   img,
                            bool                   normalize,
                            BaseColor              b,
                            int                    skip )
{
    normalizeImage( img, normalize );

    std::vector<TriplePoint> out;
    points.debug_out( maxSize, out );
    std::vector<TriplePoint>::iterator out_it  = out.begin();
    std::vector<TriplePoint>::iterator out_end = out.end();

    /* All coordinates in the TriplePoint array are plotted into the
     * image with brightness 3. */
    for( ; out_it != out_end; out_it++ ) {
        const int2& coord = out_it->coord; // array[i].coord;
        const int2& befor = out_it->descending.befor; // array[i].befor;

        if( outOfBounds( coord.x, coord.y, img ) ) {
            cout << "Coord of point (" << coord.x << "," << coord.y << ") is out of bounds (" << __LINE__ << ")" << endl;
        } else {
            // if( befor.x != 0 && befor.y != 0 && after.x != 0 && after.y != 0 )
            if( befor.x != 0 && befor.y != 0 ) {
                img.ptr(coord.y)[coord.x] = GREY2;
            }
        }

        if( out_it->_coords_idx > 1 ) {
            int coloradd;
            if( out_it->_coords_idx == 8 )
                coloradd = getColor( b );
            else
                coloradd = BLUE;
            for( int j=1; j<out_it->_coords_idx; j++ ) {
                plotOneLine( out_it->_coords[j-1],
                             out_it->_coords[j],
                             img,
                             coloradd );
            }
        }
        for( int skipct=0; skipct<skip; skipct++ ) {
            out_it++;
            if( out_it == out_end ) return;
        }
    }
}
#endif // NDEBUG

}; // namespace cctag

