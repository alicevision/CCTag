#ifndef _CCTAG_LEVEL_HPP
#define	_CCTAG_LEVEL_HPP

#include <opencv2/opencv.hpp>

namespace popart {
    class TagPipe;
};

namespace cctag {

class Parameters;

class Level
{
public:
  
  Level( std::size_t width, std::size_t height, int debug_info_level, bool cuda_allocates = false );
  
  ~Level( );

  void setLevel( const cv::Mat & src,
                 const double thrLowCanny,
                 const double thrHighCanny,
                 const cctag::Parameters* params );
  void setLevel( popart::TagPipe* cuda_pipe,
                 const cctag::Parameters& params );

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
    return _cols;
  }
  
  inline std::size_t height() const
  {
    return _rows;
  }
  

private:
  int         _level;
  bool        _cuda_allocates;
  bool        _mat_initialized_from_cuda;
  std::size_t _cols;
  std::size_t _rows;
  
  cv::Mat* _dx;
  cv::Mat* _dy;
  cv::Mat* _mag;
  cv::Mat* _src;
  cv::Mat* _edges;
  cv::Mat  _temp;
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
  cv::Mat _edgesNotThin;
#endif
};

}

#endif	/* _CCTAG_LEVEL_HPP */
