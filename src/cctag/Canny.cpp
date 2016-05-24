#include <cctag/Canny.hpp>

#include "utils/Defines.hpp"

//#define USE_CANNY_OCV3
#ifdef USE_CANNY_OCV3
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/timer.hpp>
#endif

namespace cctag
{

void edgesPointsFromCanny(
        EdgePointCollection& edgeCollection,
        const cv::Mat & edges,
        const cv::Mat & dx,
        const cv::Mat & dy )
{
  std::size_t width = edges.cols;
  std::size_t height = edges.rows;
  
  edgeCollection.set_shape(width, height);
  
  for( int y = 0 ; y < height ; ++y )
  {
    for( int x = 0 ; x < width ; ++x )
    {
      if ( edges.at<uchar>(y,x) == 255 )
      {
        edgeCollection.add_point(x, y, dx.at<short>(y,x), dy.at<short>(y,x));
      }
    }
  }
}

} // namespace cctag


