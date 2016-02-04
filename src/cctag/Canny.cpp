#include <cctag/Canny.hpp>

#include <boost/gil/image_view.hpp>

#include "Global.hpp"

//#define USE_CANNY_OCV3
#ifdef USE_CANNY_OCV3
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/timer.hpp>
#endif

namespace cctag
{

void edgesPointsFromCanny(
        std::vector<EdgePoint>& points,
        EdgePointsImage & edgePointsMap,
        const cv::Mat & edges,
        const cv::Mat & dx,
        const cv::Mat & dy )
{
  std::size_t width = edges.cols;
  std::size_t height = edges.rows;

  edgePointsMap.resize( boost::extents[width][height] );
  std::fill( edgePointsMap.origin(), edgePointsMap.origin() + edgePointsMap.size(), (EdgePoint*)NULL );
  
  points.reserve( width * height / 2 ); // todo: allocate that in the memory pool @Lilian

  for( int y = 0 ; y < height ; ++y )
  {
    for( int x = 0 ; x < width ; ++x )
    {
      if ( edges.at<uchar>(y,x) == 255 )
      {
        points.push_back( EdgePoint( x, y, (float) dx.at<short>(y,x), (float) dy.at<short>(y,x) ) );
        
        EdgePoint* p = &points.back();
        edgePointsMap[x][y] = p;
      }
    }
  }

}

} // namespace cctag


