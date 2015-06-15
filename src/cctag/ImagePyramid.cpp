// TODO TODO TODO TODO TODO
// New simplified version of pyramidImage 

#include <cctag/ImagePyramid.hpp>

#include <iostream>


namespace cctag {
  
Frame::Frame( uint32_t width, uint32_t height )
  : _dx(0)
  , _dy(0)
  , _mag(0)
  , _src(0)
  , _edges(0)
  , _width(width)
  , _height(height)
{
    std::cerr << "Allocating frame: " << width << "x" << height << std::endl;
  // Allocation
}

ImagePyramid::ImagePyramid( const unsigned char* src, std::size_t pitch, const std::size_t width, const std::size_t height, const std::size_t nbLevels )
{
  uint32_t w = width;
  uint32_t h = height;

  for(int i = 0; i < nbLevels ; ++i)
  {
    _imagePyramid[i] = new Frame( w, h );

    // TODO
    // cvResize( . , . );
    // ...

    w = ( w >> 1 ) + ( w & 1 );
    h = ( h >> 1 ) + ( h & 1 );
  }

}

Frame* ImagePyramid::getFrame( const std::size_t level ) const
{
        return _imagePyramid[level];
}

}


