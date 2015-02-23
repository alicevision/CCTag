/* bug fix for boost::gil */
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#include <iostream>
#include <limits>

#include "view.hpp"
#include "debug.hpp"

#ifdef WITH_CUDA
  #include <cuda_runtime.h>
#endif

#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/filesystem.hpp>

namespace popart {

MultiresolutionCanvas View::_canvas;

View::View( const std::string& filename )
{
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
}

View::~View( )
{ }

void View::setNumLayers( size_t num )
{
#ifdef WITH_CUDA
    if( _canvas.resizeRequired( _image.width(), _image.height() ) ) {
        _canvas.uninit( );
        _canvas.init( _image.width(), _image.height(), num );
    }

    POP_INFO << "BEGIN - PIXEL-WISE COPYING - THIS MUST BE REMOVED" << std::endl;
    size_t   host_canvas_width      = _canvas.hostCanvasWidth();
    size_t   host_canvas_layer_size = _canvas.hostCanvasSize();
    uint8_t* ptr                    = _canvas.hostCanvas();
    for( size_t y=0; y<_image.height(); y++ ) {
        for( size_t x=0; x<_image.width(); x++ ) {
            boost::gil::rgb8_pixel_t pixel = *_view.at( x, y );
            ptr[0 * host_canvas_layer_size + y * host_canvas_width + x] = pixel[0];
            ptr[1 * host_canvas_layer_size + y * host_canvas_width + x] = pixel[1];
            ptr[2 * host_canvas_layer_size + y * host_canvas_width + x] = pixel[2];
        }
    }
    POP_INFO << "END - PIXEL-WISE COPYING - THIS MUST BE REMOVED" << std::endl;

    _canvas.uploadPixels( );
#endif
}

MultiresolutionCanvas::MultiresolutionCanvas( )
    : _original_width( 0 )
    , _original_height( 0 )
    , _num_layers( 0 )
    , _canvas_size( 0 )
    , _host_image( 0 )
    , _device_all_int_canvasses( 0 )
    , _device_all_float_canvasses( 0 )
{ }

void MultiresolutionCanvas::init( size_t width, size_t height, size_t num )
{
#ifdef WITH_CUDA
    POP_ENTER;
    assert( num > 0 );

    _original_width = width;
    _original_height = height;
    _num_layers = num;
    _canvas_size = new CanvasSize[num];
    _host_image = 0;
    _device_all_int_canvasses = 0;
    _device_all_float_canvasses = 0;

    // find width at lowest resolution in pixels
    size_t base_width = ( width >> (num-1) );
    if( (base_width << (num-1)) != width ) {
        base_width += 1;
    }

    // make sure we can divide width by 8 even for smallest resolution
    if( ( base_width & 0x7 ) ) {
        base_width &= ~0x7;
        base_width +=  0x8;
    }

    // find height at lowest resolution in pixels
    size_t base_height = ( height >> (num-1) );
    if( (base_height << (num-1)) != height ) {
        base_height += 1;
    }

    for( int i=num-1; i>=0; i-- ) {
        _canvas_size[i]._width  = base_width;
        _canvas_size[i]._height = base_height;
        base_width  <<= 1;
        base_height <<= 1;
    }

    size_t host_canvas_size = _canvas_size[0].sz() * 3 * sizeof(uint8_t);
    POP_INFO << "Trying to allocate CUDA host memory. " << host_canvas_size << " (bytes)" <<  std::endl;
    cudaError_t err;
    err = cudaHostAlloc( &_host_image, host_canvas_size, cudaHostAllocDefault );
    if( err != cudaSuccess ) {
        POP_ERROR << "Failure to allocate CUDA host memory. " << cudaGetErrorString(err) << std::endl;
        _host_image = 0;
    }

    size_t gpu_space = 0;
    for( size_t i=0; i<_num_layers; i++ ) {
        gpu_space += ( _canvas_size[i].sz() );
    }

    /// Most of the time, RGB info on the host is larger than Y info of all resolutions on the GPU
    gpu_space = std::max<size_t>(gpu_space,host_canvas_size);

    POP_INFO << "Trying to allocate CUDA device memory (int). " << gpu_space << " bytes" << std::endl;
    err = cudaMalloc( &_device_all_int_canvasses, gpu_space * sizeof(uint8_t) );
    if( err != cudaSuccess ) {
        POP_ERROR << "Failure to allocate CUDA device memory (int). " << cudaGetErrorString(err) << std::endl;
        _device_all_int_canvasses = 0;
    }

    POP_INFO << "Trying to allocate CUDA device memory (float). " << gpu_space*sizeof(float) << " bytes" << std::endl;
    err = cudaMalloc( &_device_all_float_canvasses, gpu_space * sizeof(float) );
    if( err != cudaSuccess ) {
        POP_ERROR << "Failure to allocate CUDA device memory (float). " << cudaGetErrorString(err) << std::endl;
        _device_all_float_canvasses = 0;
    }
    POP_LEAVE;
#endif
}

MultiresolutionCanvas::~MultiresolutionCanvas( )
{
    uninit( );
}

void MultiresolutionCanvas::uninit( )
{
#ifdef WITH_CUDA
    POP_INFO << "deallocating CUDA memory" << std::endl;
    if( _device_all_float_canvasses ) cudaFree( _device_all_float_canvasses );
    if( _device_all_int_canvasses )   cudaFree( _device_all_int_canvasses );
    if( _host_image )                 cudaFreeHost( _host_image );
    _device_all_float_canvasses = 0;
    _device_all_int_canvasses = 0;
    _host_image = 0;

    if( _canvas_size ) delete [] _canvas_size;
    _canvas_size = 0;
#endif
}

bool MultiresolutionCanvas::resizeRequired( size_t width, size_t height ) const
{
    return ( ( width != _original_width ) || ( height != _original_height ) );
}

void MultiresolutionCanvas::uploadPixels( )
{
#ifdef WITH_CUDA
    /// precondition: everything allocated, sizes checked
    cudaError_t err;
    err = cudaMemcpy( _device_all_int_canvasses, _host_image, _canvas_size[0].sz()*3*sizeof(uint8_t), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        POP_ERROR << "Uploading pixels to GPU failed. " << cudaGetErrorString(err) << std::endl;
    }
#endif
}

} // namespace popart

