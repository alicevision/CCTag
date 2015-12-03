#ifndef _CCTAG_VIEW_HPP_
#define _CCTAG_VIEW_HPP_

#pragma once

#include <string>
#include <vector>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

namespace cctag {

class View
{
public:
    boost::gil::rgb8_image_t _image;
    boost::gil::rgb8_view_t  _view;
    
    // Gray image
    boost::gil::gray8_image_t _grayImage;
    boost::gil::gray8_view_t  _grayView;

    // static MultiresolutionCanvas _canvas;

public:
    View( const std::string& filename );
    View( const unsigned char * rawData, size_t width, size_t height, ptrdiff_t src_row_bytes );
    
    ~View( );

    void setNumLayers( size_t num );
};

} // namespace cctag

#endif