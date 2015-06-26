#include <cctag/global.hpp>
#include <cctag/ImagePyramid.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/filter/thinning.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

namespace cctag {
  
Level::Level( std::size_t width, std::size_t height )
{
  // Allocation
  _src = cv::Mat(height, width, CV_8UC1);
  _dx =  cv::Mat(height, width, CV_16SC1 );
  _dy = cv::Mat(height, width, CV_16SC1 );
  _mag = cv::Mat(height, width, CV_16SC1 );
  _edges = cv::Mat(height, width, CV_8UC1);
  _temp = cv::Mat(height, width, CV_8UC1);
}

void Level::setLevel( const cv::Mat & src )
{
  cv::resize(src, _src, cv::Size(_src.cols,_src.rows));
  // ASSERT TODO : check that the data are allocated here
  // Compute derivative and canny edge extraction.
  cvRecodedCanny(_src,_edges,_dx,_dy,0, 30, 3 | CV_CANNY_L2_GRADIENT );
  // Perform the thinning.
  thin(_edges,_temp);
}

const cv::Mat & Level::getSrc() const
{
  return _src;
}

const cv::Mat & Level::getDx() const
{
  return _dx;
}

const cv::Mat & Level::getDy() const
{
  return _dy;
}

const cv::Mat & Level::getMag() const
{
  return _mag;
}

const cv::Mat & Level::getEdges() const
{
  return _edges;
}

ImagePyramid::ImagePyramid()
{
}

ImagePyramid::ImagePyramid( std::size_t width, std::size_t height, const std::size_t nLevels )
{
  _levels.clear();
  for(int i = 0; i < nLevels ; ++i)
  {
    _levels.push_back(new Level( width, height ));
    width /= 2;
    height /= 2;
  }
}

void ImagePyramid::build( const cv::Mat & src )
{
  _levels[0]->setLevel( src );
  
  for(int i = 1; i < _levels.size() ; ++i)
  {
    _levels[i]->setLevel( _levels[i-1]->getSrc() );
  }
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

}


