/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <exception>
#include <iomanip>
#include <random>

void generateCompressedFrame(const cv::Mat & src, cv::Mat & dst, std::default_random_engine & generator )
{
  //cv::imwrite(outputSrc, src);
  cv::Mat tmp;
  cv::medianBlur(src, tmp, 5);
  //cv::imwrite(outputFilt, tmp);
  
  // Define random generator with Gaussian distribution
  const float mean = 0.0;
  const float stddev = 1.5;
  std::normal_distribution<float> dist(mean, stddev);

  //cv::Mat tmp;
  //src.copyTo(tmp);

  // Add Gaussian noise
  for(std::size_t i=1 ; i<src.rows-1 ; i=i+3){
    for(std::size_t j=1 ; j<src.cols-1 ; j=j+3){
      int noise = (int) dist(generator);
      for(std::size_t n=0 ; n<3 ; ++n){
        for(std::size_t m=0 ; m<3 ; ++m){
          std::size_t y = i-1+n;
          std::size_t x = j-1+m;
          int value = ((int) tmp.at<uchar>(y,x) + noise > 255 ) ? 255 : (int) tmp.at<uchar>(y,x) + noise;
          value = (value < 0 ) ? 0 : value;
          tmp.at<uchar>(y,x) = (uchar) value;
        }
      }
    }
  }

  for(std::size_t i=0 ; i<src.rows ; ++i){
    for(std::size_t j=0 ; j<src.cols ; ++j){
      int noise = (int) dist(generator);
      std::size_t y = i;
      std::size_t x = j;
      int value = ((int) tmp.at<uchar>(y,x) + noise > 255 ) ? 255 : (int) tmp.at<uchar>(y,x) + noise;
      value = (value < 0 ) ? 0 : value;
      tmp.at<uchar>(y,x) = (uchar) value;
    }
  }
  
  dst = tmp;
}

/*************************************************************/
/*                    Main entry                             */
/*************************************************************/
int main(int argc, char** argv)
{
  std::size_t nFrames = 100;
  
    // Gray scale convertion
  cv::Mat src = cv::imread(argv[1]);
  cv::Mat graySrc;
  cv::cvtColor( src, graySrc, CV_BGR2GRAY );
  
  // Initialize random generator
  std::default_random_engine generator;
  
  for(std::size_t i=0 ; i<nFrames ; ++i)
  {
    std::stringstream outFileName;
    outFileName << std::setfill('0') << std::setw(5) << i;
    
    // Generate image.
    cv::Mat dst;
    generateCompressedFrame(graySrc, dst, generator);
    
    // Write generated noisy image.
    cv::imwrite(std::string(argv[2]) + "/" + outFileName.str()+".png", dst);
  }
    
  return 0;
}

