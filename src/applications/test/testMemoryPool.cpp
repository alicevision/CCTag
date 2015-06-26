#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/global.hpp>
#include <cctag/filter/cannyOpenCV3/cvRecodeCannyOCV3.hpp>
#include <cctag/filter/cvRecode.hpp>

#include "opencv2/opencv.hpp"
#include <boost/timer/timer.hpp>

#include <cstdlib>

//#define OPENCV3

using namespace cv;
using namespace boost::posix_time;

int main(int argc, char** argv)
{
  cctag::MemoryPool::instance().updateMemoryAuthorizedWithRAM();
  
  VideoCapture cap(argv[1]); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
      return -1;

  std::size_t width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  std::size_t height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  
  CCTAG_COUT_VAR(width);
  CCTAG_COUT_VAR(height);
  
  ptime t1(microsec_clock::local_time());
  
  const std::size_t nLevels = 4;
  
  cctag::MemoryPool::instance().allocateImagePyramid(width, height, nLevels);
  
  ptime t2(microsec_clock::local_time());
  time_duration d = t2 - t1;
  const double spendTime = d.total_milliseconds();
  CCTAG_COUT("Allocation took:" << spendTime << " ms");
  
  cv::Mat imgCanny(height, width, CV_8UC1);
#ifndef OPENCV3
  cv::Mat imgDX(height, width, CV_16SC1 );
  cv::Mat imgDY(height, width, CV_16SC1 );
#endif
  
  Mat edges;
  namedWindow("edges",1);
  for(;;)
  {
      Mat frame;
      cap >> frame; // get a new frame from camera

      cvtColor(frame, edges, CV_BGR2GRAY);
      
      ptime tstop1(microsec_clock::local_time());
      
      cctag::MemoryPool::instance().getImagePyramid().build(edges);
      
      cctag::MemoryPool::instance().getImagePyramid().output();

//#ifndef OPENCV3
//      // OpenCV 2.4.9 (recoded, kernel 9x9)
//      cvRecodedCanny(edges,imgCanny,imgDX,imgDY,0, 30, 3 | CV_CANNY_L2_GRADIENT );
//#else
//      // OpenCV 3.0.0
//      RecodedCanny(edges, imgCanny, 0, 30, 3);
//#endif
      
      ptime tstop2(microsec_clock::local_time());
      
      time_duration d2 = tstop2 - tstop1;
      const double spendTime2 = d2.total_milliseconds();
      CCTAG_COUT_OPTIM("Canny took:" << spendTime2 << " ms");
      
      imshow("edges", cctag::MemoryPool::instance().getImagePyramid().getLevel(2)->getEdges());//cctag::MemoryPool::instance().getCannyImage());
      if(waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}

