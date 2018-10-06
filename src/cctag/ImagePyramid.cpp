/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/Defines.hpp>
#include <cctag/ImagePyramid.hpp>
#include <cctag/utils/VisualDebug.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

namespace cctag {

ImagePyramid::ImagePyramid()
{
}

ImagePyramid::ImagePyramid( std::size_t width, std::size_t height, std::size_t nLevels, bool cuda_allocates )
{
  _levels.clear();
  _levels.resize(nLevels);
  for(int i = 0; i < nLevels ; ++i)
  {
    _levels[i] = new Level( width, height, i, cuda_allocates );
    width /= 2;
    height /= 2;
  }
}

void ImagePyramid::build( const cv::Mat & src, float thrLowCanny, float thrHighCanny, const cctag::Parameters* params )
{
#ifdef CCTAG_WITH_CUDA
    if( params->_useCuda ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    must not call " << __FUNCTION__ << " with CUDA enables" << std::endl;
        exit( -1 );
    }
#endif // CCTAG_WITH_CUDA

    /* The pyramid building function is never called if CUDA is used.
     */
  _levels[0]->setLevel( src , thrLowCanny, thrHighCanny, params );
  
  for(int i = 1; i < _levels.size() ; ++i)
  {
    _levels[i]->setLevel( _levels[i-1]->getSrc(), thrLowCanny, thrHighCanny, params );
  }
  
#ifdef CCTAG_SERIALIZE
  for(int i = 0; i < _levels.size() ; ++i)
  {
    std::stringstream outFilenameCanny;
    outFilenameCanny << "cannyLevel" << i;
    CCTagVisualDebug::instance().initBackgroundImage(_levels[i]->getEdges());
    CCTagVisualDebug::instance().newSession(outFilenameCanny.str());
    
#ifdef CCTAG_EXTRA_LAYER_DEBUG
    std::stringstream dX, dY;
    cv::Mat imgDX, imgDY;
    
    dX << "dX" << i;
    sIntToUchar(_levels[i]->getDx(), imgDX);
    CCTagVisualDebug::instance().initBackgroundImage(imgDX);
    CCTagVisualDebug::instance().newSession(dX.str());   
    dY << "dY" << i;
    sIntToUchar(_levels[i]->getDy(), imgDY);
    CCTagVisualDebug::instance().initBackgroundImage(imgDY);
    CCTagVisualDebug::instance().newSession(dY.str()); 
    
    outFilenameCanny << "_wt";
    CCTagVisualDebug::instance().initBackgroundImage(_levels[i]->getCannyNotThin());
    CCTagVisualDebug::instance().newSession(outFilenameCanny.str());
    
    CCTAG_COUT("src_");
    CCTagVisualDebug::instance().coutImage<uchar>(_levels[i]->getSrc());
    
    CCTAG_COUT("dx_");
    CCTagVisualDebug::instance().coutImage<short>(_levels[i]->getDx());
    CCTAG_COUT("dy_");
    CCTagVisualDebug::instance().coutImage<short>(_levels[i]->getDy());
#endif
  }
#endif
}

ImagePyramid::~ImagePyramid()
{
  for(auto & _level : _levels)
  {
    delete _level;
  }
};

std::size_t ImagePyramid::getNbLevels() const
{
  return _levels.size();
}

Level* ImagePyramid::getLevel( std::size_t level ) const
{
        return _levels[level];
}

void toUchar(const cv::Mat & src, cv::Mat & dst)
{
  std::size_t width = src.cols;
  std::size_t height = src.rows;
  dst = cv::Mat(height, width, CV_8UC1);
  
  double min = 0;
  double max = 0;
  
  cv::minMaxLoc(src, &min, &max);
  
  CCTAG_COUT_VAR(min);
  CCTAG_COUT_VAR(max);
  
  float scale = 255/(max-min);
  
  for ( int i=0 ; i < width ; ++i)
  {
    for ( int j=0 ; j < height ; ++j)
    {
      dst.at<uchar>(j,i) = (uchar) ((src.at<short>(j,i)+min)*scale);
    }
  }
}

}


