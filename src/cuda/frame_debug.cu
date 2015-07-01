#include <iostream>
#include <limits>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include "debug_macros.hpp"

#include "../cctag/cmdline.hpp"

#include "frame.h"
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
    delete [] _h_debug_edgelist;
    delete [] _h_debug_edgelist_2;

    _h_debug_plane      = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_smooth     = new float[ getWidth() * getHeight() ];
    _h_debug_dx         = new int16_t[ getWidth() * getHeight() ];
    _h_debug_dy         = new int16_t[ getWidth() * getHeight() ];
    _h_debug_mag        = new uint32_t[ getWidth() * getHeight() ];
    _h_debug_map        = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_hyst_edges = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_edges      = new unsigned char[ getWidth() * getHeight() ];
    _h_debug_edgelist   = new int2[ min(params._maxEdges,_h_edgelist_sz) ];
    _h_debug_edgelist_2 = new TriplePoint[ min(params._maxEdges,_h_edgelist_2_sz) ];

    POP_SYNC_CHK;

#if 0
    cerr << "Trigger download of debug plane: "
         << "(" << _d_plane.cols << "," << _d_plane.rows << ") pitch " << _d_plane.step
         << " to "
         << "(" << getWidth() << "," << getHeight() << ")" << endl;
#endif
    POP_CUDA_MEMCPY_2D_ASYNC( _h_debug_plane, getWidth(),
                              _d_plane.data, _d_plane.step,
                              _d_plane.cols,
                              _d_plane.rows,
                              cudaMemcpyDeviceToHost, _stream );
#if 0
    cerr << "Trigger download of Gaussian debug plane: "
         << "(" << _d_smooth.cols << "," << _d_smooth.rows << ") pitch " << _d_smooth.step
         << " to "
         << "(" << getWidth() << "," << getHeight() << ")" << endl;
#endif
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
    if( _h_edgelist_sz > 0 ) {
        POP_CUDA_MEMCPY_ASYNC( _h_debug_edgelist,
                               _d_edgelist,
                               min(params._maxEdges,_h_edgelist_sz) * sizeof(int2),
                               cudaMemcpyDeviceToHost, _stream );
    }
    if( _h_edgelist_2_sz > 0 ) {
        POP_CUDA_MEMCPY_ASYNC( _h_debug_edgelist_2,
                               _d_edgelist_2,
                               min(params._maxEdges,_h_edgelist_2_sz) * sizeof(TriplePoint),
                               cudaMemcpyDeviceToHost, _stream );
    }
}

#if 0
__host__
static void testme( cv::cuda::PtrStepSzf src )
{
    size_t non_null_ct = 0;
    float minval = 1000.0f;
    float maxval = -1000.0f;
    for( size_t i=0; i<src.rows; i++ )
        for( size_t j=0; j<src.cols; j++ ) {
            float f = src.ptr(i)[j];
            if( f != 0.0f )
                non_null_ct++;
            minval = min( minval, f );
            maxval = max( maxval, f );
        }
    printf("testme: There are %lu non-null values in the Gaussian end result (min %f, max %f)\n", (unsigned long)non_null_ct, minval, maxval );
}
#endif

void Frame::writeDebugPlane1( const char* filename, const cv::cuda::PtrStepSzb& plane )
{
#ifdef CHATTY_WRITE_DEBUG_PLANE
    cerr << "Enter " << __FUNCTION__ << endl;
#endif
    assert( plane.data );

    ofstream of( filename );
    of << "P5" << endl
       << plane.cols << " " << plane.rows << endl
       << "255" << endl;
    of.write( (char*)plane.data, plane.cols * plane.rows );
#ifdef CHATTY_WRITE_DEBUG_PLANE
    cerr << "Leave " << __FUNCTION__ << endl;
#endif
}

template<class T>
__host__
void Frame::writeDebugPlane( const char* filename, const cv::cuda::PtrStepSz<T>& plane )
{
#ifdef CHATTY_WRITE_DEBUG_PLANE
    cerr << "Enter " << __FUNCTION__ << endl;
    cerr << "    filename: " << filename << endl;
#endif

    ofstream of( filename );
    of << "P5" << endl
       << plane.cols << " " << plane.rows << endl
       << "255" << endl;

    // T minval = 1000;  // std::numeric_limits<float>::max();
    // T maxval = -1000; // std::numeric_limits<float>::min();
    T minval = std::numeric_limits<T>::max();
    T maxval = std::numeric_limits<T>::min();
    // for( uint32_t i=0; i<plane.rows*plane.cols; i++ ) {
    for( size_t i=0; i<plane.rows; i++ ) {
        for( size_t j=0; j<plane.cols; j++ ) {
            T f = plane.ptr(i)[j];
            // float f = plane.data[i];
            minval = min( minval, f );
            maxval = max( maxval, f );
        }
    }
#ifdef CHATTY_WRITE_DEBUG_PLANE
    cerr << "    step size is " << plane.step << endl;
    cerr << "    found minimum value " << minval << endl;
    cerr << "    found maximum value " << maxval << endl;
#endif

    // testme( plane );

    float fmaxval = 255.0 / ( (float)maxval - (float)minval );
    for( uint32_t i=0; i<plane.rows*plane.cols; i++ ) {
        T f = plane.data[i];
        float outf = ( (float)f - (float)minval ) * fmaxval;
        unsigned char uc = (unsigned char)outf;
        of << uc;
    }

#ifdef CHATTY_WRITE_DEBUG_PLANE
    cerr << "Leave " << __FUNCTION__ << endl;
#endif
}

void Frame::writeInt2Array( const char* filename, const int2* array, uint32_t sz )
{
    ofstream of( filename );

    for( uint32_t i=0; i<sz; i++ ) {
        of << array[i].x << " " << array[i].y << endl;
    }
}

void Frame::writeTriplePointArray( const char* filename, const TriplePoint* array, uint32_t sz )
{
    ofstream of( filename );

    for( uint32_t i=0; i<sz; i++ ) {
        of << array[i].coord.x << " " << array[i].coord.y << " "
           << array[i].befor.x << " " << array[i].befor.y << " "
           << array[i].after.x << " " << array[i].after.y << endl;
    }
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

void Frame::debugPlotPointsIntoImage( const TriplePoint* array, uint32_t sz, cv::cuda::PtrStepSzb img )
{
    for( uint32_t x=0; x<img.cols; x++ ) {
        for( uint32_t y=0; y<img.rows; y++ ) {
            if( img.ptr(y)[x] != 0 ) img.ptr(y)[x] = 1;
        }
    }
    for( uint32_t i=0; i<sz; i++ ) {
        const int2& coord = array[i].coord;
        const int2& befor = array[i].befor;
        // const int2& after = array[i].after;
        if( outOfBounds( coord.x, coord.y, img ) ) {
            cout << "Coord of point (" << coord.x << "," << coord.y << ") is out of bounds" << endl;
        } else {
            // if( befor.x != 0 && befor.y != 0 && after.x != 0 && after.y != 0 )
            if( befor.x != 0 && befor.y != 0 ) {
                img.ptr(coord.y)[coord.x] = 3;
            }
        }
    }
}

void Frame::writeHostDebugPlane( string filename, const cctag::Parameters& params )
{
    struct stat st = {0};

    string dir = cmdline.debugDir;
    char   dirtail = dir[ dir.size()-1 ];
    if( dirtail != '/' ) {
        filename = dir + "/" + filename;
    } else {
        filename = dir + filename;
    }

    if (stat( dir.c_str(), &st) == -1) {
        mkdir( dir.c_str(), 0700);
    }

    string s = filename + ".pgm";
    cv::cuda::PtrStepSzb b( getHeight(),
                            getWidth(),
                            _h_debug_plane,
                            getWidth() );
    writeDebugPlane1( s.c_str(), b );

    {
        ofstream of( ( filename + "-img-ascii.txt" ).c_str() );
        for( int y=0; y<getHeight(); y++ ) {
            for( int x=0; x<getWidth(); x++ )
            {
                int val = b.ptr(y)[x];
                of << val << " ";
            }
            of << endl;
        }
    }

    s = filename + "-gauss.pgm";
    cv::cuda::PtrStepSzf smooth( getHeight(),
                                 getWidth(),
                                 _h_debug_smooth,
                                 getWidth()*sizeof(float) );
    writeDebugPlane( s.c_str(), smooth );

    {
        ofstream of( ( filename + "-gauss-ascii.txt" ).c_str() );
        for( int y=0; y<getHeight(); y++ ) {
            for( int x=0; x<getWidth(); x++ )
            {
                int val = smooth.ptr(y)[x];
                of << val << " ";
            }
            of << endl;
        }
    }

    s = filename + "-dx.pgm";
    cv::cuda::PtrStepSz16s dx( getHeight(),
                               getWidth(),
                               _h_debug_dx,
                               getWidth()*sizeof(int16_t) );
    writeDebugPlane( s.c_str(), dx );

    {
        ofstream of( ( filename + "-dx-ascii.txt" ).c_str() );
        for( int y=0; y<getHeight(); y++ ) {
            for( int x=0; x<getWidth(); x++ )
            {
                int val = dx.ptr(y)[x];
                of << val << " ";
            }
            of << endl;
        }
    }

    s = filename + "-dy.pgm";
    cv::cuda::PtrStepSz16s dy( getHeight(),
                               getWidth(),
                               _h_debug_dy,
                               getWidth()*sizeof(int16_t) );
    writeDebugPlane( s.c_str(), dy );

    {
        ofstream of( ( filename + "-dy-ascii.txt" ).c_str() );
        for( int y=0; y<getHeight(); y++ ) {
            for( int x=0; x<getWidth(); x++ )
            {
                int val = dy.ptr(y)[x];
                of << val << " ";
            }
            of << endl;
        }
    }

    s = filename + "-mag.pgm";
    cv::cuda::PtrStepSz32u mag( getHeight(),
                                getWidth(),
                                _h_debug_mag,
                                getWidth()*sizeof(uint32_t) );
    writeDebugPlane( s.c_str(), mag );

    s = filename + "-map.pgm";
    cv::cuda::PtrStepSzb   map( getHeight(),
                                getWidth(),
                                _h_debug_map,
                                getWidth()*sizeof(uint8_t) );
    writeDebugPlane( s.c_str(), map );

    {
        ofstream of( ( filename + "-map-ascii.txt" ).c_str() );
        for( int y=0; y<getHeight(); y++ ) {
            for( int x=0; x<getWidth(); x++ )
            {
                int val = map.ptr(y)[x];
                of << val << " ";
            }
            of << endl;
        }
    }

    s = filename + "-hystedges.pgm";
    cv::cuda::PtrStepSzb   hystedges( getHeight(),
                                      getWidth(),
                                      _h_debug_hyst_edges,
                                      getWidth()*sizeof(uint8_t) );
    writeDebugPlane( s.c_str(), hystedges );

    s = filename + "-edges.pgm";
    cv::cuda::PtrStepSzb   edges( getHeight(),
                                  getWidth(),
                                  _h_debug_edges,
                                  getWidth()*sizeof(uint8_t) );
    writeDebugPlane( s.c_str(), edges );

    if( _h_edgelist_sz > 0 ) {
        s = filename + "-edgelist.txt";
        writeInt2Array( s.c_str(), _h_debug_edgelist, min(params._maxEdges,_h_edgelist_sz) );
    }

    if( _h_edgelist_2_sz > 0 ) {
        s = filename + "-edgelist2.txt";
        writeTriplePointArray( s.c_str(), _h_debug_edgelist_2, min(params._maxEdges,_h_edgelist_2_sz) );

        debugPlotPointsIntoImage( _h_debug_edgelist_2, min(params._maxEdges,_h_edgelist_2_sz), edges );

        s = filename + "-edges-dots.pgm";
        writeDebugPlane( s.c_str(), edges );
    }
}

}; // namespace popart

