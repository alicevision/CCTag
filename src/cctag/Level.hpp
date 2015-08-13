#ifndef _CCTAG_LEVEL_HPP
#define	_CCTAG_LEVEL_HPP

#include <opencv2/opencv.hpp>

namespace cctag {

class Level
{
public:
  
  Level( std::size_t width, std::size_t height );
  
  void setLevel(const cv::Mat & src, const double thrLowCanny, const double thrHighCanny);
  const cv::Mat & getSrc() const;
  const cv::Mat & getDx() const;
  const cv::Mat & getDy() const;
  const cv::Mat & getMag() const; 
  const cv::Mat & getEdges() const;
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
  const cv::Mat & getCannyNotThin() const;
#endif
  
  inline std::size_t width() const
  {
    return _src.cols;
  }
  
  inline std::size_t height() const
  {
    return _src.rows;
  }
  

private:
  
  cv::Mat _dx;
  cv::Mat _dy;
  cv::Mat _mag;
  cv::Mat _src;
  cv::Mat _edges;
  cv::Mat _temp;
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
  cv::Mat _edgesNotThin;
#endif
};

}

#endif	/* _CCTAG_LEVEL_HPP */