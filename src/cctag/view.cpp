/* bug fix for boost::gil */
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#include <iostream>
#include <limits>

#include <cctag/view.hpp>
#include <cctag/debug.hpp>
#include <cctag/image.hpp>

#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/filesystem.hpp>

namespace cctag {

View::View( const std::string& filename )
{
    std::cerr << "Enter " << __FUNCTION__ << std::endl;
    std::string ext = boost::filesystem::path(filename).extension().string();

    if (ext == ".png") {
        _image = boost::gil::rgb8_image_t(boost::gil::png_read_dimensions(filename.c_str()));
        _view  = boost::gil::rgb8_view_t(view(_image));
        boost::gil::png_read_and_convert_view(filename.c_str(), _view);
    } else if (ext == ".jpg")  {
        _image = boost::gil::rgb8_image_t(boost::gil::jpeg_read_dimensions(filename.c_str()));
        _view  = boost::gil::rgb8_view_t(view(_image));
        boost::gil::jpeg_read_and_convert_view(filename.c_str(), _view);
    } else {
        std::cerr << __FILE__ << ":" << __LINE__ << " Programming error: unexpected file extension"
                  << std::endl;
        exit( -__LINE__ );
    }

    // Grayscale transform
    _grayView = cctag::img::toGray(_view, _grayImage);
    std::cerr << "Leave " << __FUNCTION__ << std::endl;
}

View::View( const unsigned char * rawData, size_t width, size_t height, ptrdiff_t src_row_bytes )
{
    _grayView = boost::gil::interleaved_view(width, height, (boost::gil::gray8_pixel_t*) rawData, src_row_bytes);
    
    //boost::gil::png_write_view("/home/lilian/data/toto.png", boost::gil::color_converted_view<boost::gil::rgb8_pixel_t>(_grayView) );
}

View::~View( )
{ }

void View::setNumLayers( size_t num )
{
}

} // namespace cctag

