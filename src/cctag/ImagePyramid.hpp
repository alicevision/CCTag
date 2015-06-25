#ifndef IMAGEPYRAMID_HPP
#define	IMAGEPYRAMID_HPP

#include <opencv2/opencv.hpp>

#include <stdint.h>
#include <cstddef>
#include <vector>

// TODO TODO TODO TODO TODO
// New simplified version of pyramidImage 

namespace cctag {

class Level
{
public:
  
  Level( std::size_t width, std::size_t height );
  
  void setLevel(const cv::Mat & src);
  const cv::Mat & getSrc() const;
  const cv::Mat & getDx() const;
  const cv::Mat & getDy() const;
  const cv::Mat & getMag() const; 
  const cv::Mat & getEdges() const;

private:
  
  cv::Mat _dx;
  cv::Mat _dy;
  cv::Mat _mag;
  cv::Mat _src;
  cv::Mat _edges;

};

class ImagePyramid
{
public:
  ImagePyramid();
  
  ImagePyramid( const std::size_t width, const std::size_t height, const std::size_t nLevels );
  
  ~ImagePyramid();

  Level* getLevel( const std::size_t level ) const;
  
  std::size_t getNbLevels() const;
  
  void build(const cv::Mat & src);
  void output();

private:
  std::vector<Level*> _levels;
};

}

#endif	/* IMAGEPYRAMID_HPP */

