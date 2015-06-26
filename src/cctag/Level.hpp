#ifndef _CCTAG_LEVEL_HPP
#define	_CCTAG_LEVEL_HPP

#include <opencv2/opencv.hpp>

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
  cv::Mat _temp;
};

#endif	/* _CCTAG_LEVEL_HPP */

}
