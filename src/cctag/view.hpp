#ifndef _CCTAG_VIEW_HPP_
#define _CCTAG_VIEW_HPP_

#pragma once

#include <string>
#include <vector>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

namespace cctag {

class MultiresolutionCanvas
{
    struct CanvasSize {
        size_t _width;
        size_t _height;
        CanvasSize( ) { }
        CanvasSize( size_t x, size_t y ) : _width(x), _height(y) { }

        inline size_t sz() const {
            return _width*_height;
        }
    };

    size_t _original_width;
    size_t _original_height;
    size_t _num_layers;

    /** The rounded sizes for image planes in GPU memory.
     *  The original size is stored in _image
     */
    CanvasSize* _canvas_size;

    uint8_t* _host_image;
    uint8_t* _device_all_int_canvasses;
    float*   _device_all_float_canvasses;
public:
    MultiresolutionCanvas( );
    ~MultiresolutionCanvas( );

    void init( size_t width, size_t height, size_t layers );
    void uninit( );

    bool resizeRequired( size_t width, size_t height ) const;

    inline size_t   hostCanvasWidth() const { return _canvas_size[0]._width; }
    inline size_t   hostCanvasSize()  const { return _canvas_size[0].sz(); }
    inline uint8_t* hostCanvas()            { return _host_image; }

    void uploadPixels( );
};

class View
{
public:
    boost::gil::rgb8_image_t _image;
    boost::gil::rgb8_view_t  _view;
    
    // Gray image
    boost::gil::gray8_image_t _grayImage;
    boost::gil::gray8_view_t _grayView;

    static MultiresolutionCanvas _canvas;

public:
    View( const std::string& filename );
    View( const unsigned char * rawData, size_t width, size_t height, ptrdiff_t src_row_bytes );
    
    ~View( );

    void setNumLayers( size_t num );
};

} // namespace cctag

#endif