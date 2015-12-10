#include "cuda/framepackage.h"

// #include "cctag/fileDebug.hpp"
// #include "cctag/visualDebug.hpp"
// #include "cctag/progBase/exceptions.hpp"
// #include "cctag/detection.hpp"
// #include "cctag/view.hpp"
// #include "cctag/image.hpp"
// #include "cctag/cmdline.hpp"

// #include <boost/filesystem/convenience.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/progress.hpp>
// #include <boost/exception/all.hpp>
// #include <boost/ptr_container/ptr_list.hpp>
// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>
// #include <boost/thread/thread.hpp>
// #include <boost/thread/mutex.hpp>

#include <stdlib.h>
// #include <iostream>
#include <sys/mman.h>

// using namespace cctag;
// using boost::timer;

// using namespace boost::gil;
// namespace bfs = boost::filesystem;

namespace popart
{

FramePackage::FramePackage( int width, int height )
    : _w( width )
    , _h( height )
{
    _h_plane.data = new uint8_t[ _w * _h * sizeof(uint8_t)];
    _h_plane.step = _w * sizeof(uint8_t);
    _h_plane.cols = _w;
    _h_plane.rows = _h;

    _h_dx.data = new int16_t[ _w * _h * sizeof(int16_t)];
    _h_dx.step = _w * sizeof(int16_t);
    _h_dx.cols = _w;
    _h_dx.rows = _h;

    _h_dy.data = new int16_t[ _w * _h * sizeof(int16_t)];
    _h_dy.step = _w * sizeof(int16_t);
    _h_dy.cols = _w;
    _h_dy.rows = _h;

    _h_mag.data = new uint32_t[ _w * _h * sizeof(int32_t)];
    _h_mag.step = _w * sizeof(uint32_t);
    _h_mag.cols = _w;
    _h_mag.rows = _h;

    _h_edges.data = new uint8_t[ _w * _h * sizeof(uint8_t)];
    _h_edges.step = _w * sizeof(uint8_t);
    _h_edges.cols = _w;
    _h_edges.rows = _h;
}

FramePackage::~FramePackage( )
{
    delete [] _h_plane.data;
    delete [] _h_dx.data;
    delete [] _h_dy.data;
    delete [] _h_mag.data;
    delete [] _h_edges.data;
}

void FramePackage::pin( )
{
    mlock( _h_plane.data, _w * _h * sizeof(uint8_t) );
    mlock( _h_dx.data,    _w * _h * sizeof(int16_t) );
    mlock( _h_dy.data,    _w * _h * sizeof(int16_t) );
    mlock( _h_mag.data,   _w * _h * sizeof(int32_t) );
    mlock( _h_edges.data, _w * _h * sizeof(uint8_t) );
}

void FramePackage::unpin( )
{
    munlock( _h_plane.data, _w * _h * sizeof(uint8_t) );
    munlock( _h_dx.data,    _w * _h * sizeof(int16_t) );
    munlock( _h_dy.data,    _w * _h * sizeof(int16_t) );
    munlock( _h_mag.data,   _w * _h * sizeof(int32_t) );
    munlock( _h_edges.data, _w * _h * sizeof(uint8_t) );
}

} //namespace popart
