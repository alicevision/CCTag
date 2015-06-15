#ifndef IMAGEPYRAMID_HPP
#define	IMAGEPYRAMID_HPP

#include <stdint.h>
#include <cstddef>
#include <vector>

// TODO TODO TODO TODO TODO
// New simplified version of pyramidImage 

namespace cctag {

class Frame
{
public:
  
  Frame( uint32_t width, uint32_t height );

private:
  
  // Should match with cuda/frame.h
  int16_t*       _dx;
  int16_t*       _dy;
  uint32_t*      _mag;
  unsigned char* _src;
  unsigned char* _edges;

  std::size_t _width;
  std::size_t _height;
  std::size_t _pitch;

};

class ImagePyramid
{
  ImagePyramid( const unsigned char* src, std::size_t pitch, const std::size_t width, const std::size_t height, const std::size_t nbLevels );

  Frame* getFrame( const std::size_t level ) const;

private:
	std::vector<Frame*> _imagePyramid;
};

}

#endif	/* IMAGEPYRAMID_HPP */

