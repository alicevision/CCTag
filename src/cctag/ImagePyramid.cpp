#include <cctag/global.hpp>
#include <cctag/ImagePyramid.hpp>
#include <cctag/visualDebug.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

namespace cctag {

ImagePyramid::ImagePyramid()
{
}

ImagePyramid::ImagePyramid( std::size_t width, std::size_t height, const std::size_t nLevels )
{
  _levels.clear();
  for(int i = 0; i < nLevels ; ++i)
  {
    _levels.push_back(new Level( width, height, i ));
    width /= 2;
    height /= 2;
  }
}

void ImagePyramid::build( const cv::Mat & src, const double thrLowCanny, const double thrHighCanny, const cctag::Parameters* params )
{
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

void ImagePyramid::output()
{
  for(int i = 0; i < _levels.size() ; ++i)
  {
// todo@Lilian
    std::string basename("/home/lilian/data/");
    std::stringstream sSrc, sDx, sDy, sEdges;
    sSrc << basename << "src_" << i << ".png";
    CCTAG_COUT(sSrc.str());
    imwrite(sSrc.str(), _levels[i]->getSrc());
    sDx << basename << "dx_" << i << ".png";
    imwrite(sDx.str(), _levels[i]->getDx());
    sDy << basename << "dy_" << i << ".png";
    imwrite(sDy.str(), _levels[i]->getDy());
    sEdges << basename << "edges_" << i << ".png";
    imwrite(sEdges.str(), _levels[i]->getEdges());
  }
}

ImagePyramid::~ImagePyramid()
{
  for(int i = 0; i < _levels.size() ; ++i)
  {
    delete _levels[i];
  }
};

std::size_t ImagePyramid::getNbLevels() const
{
  return _levels.size();
}

Level* ImagePyramid::getLevel( const std::size_t level ) const
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
  
  double scale = 255/(max-min);
  
  for ( int i=0 ; i < width ; ++i)
  {
    for ( int j=0 ; j < height ; ++j)
    {
      dst.at<uchar>(j,i) = (uchar) ((src.at<short>(j,i)+min)*scale);
    }
  }
}

}


