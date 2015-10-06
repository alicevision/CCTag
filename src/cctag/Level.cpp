#include <cctag/Level.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/filter/thinning.hpp>
#include "cctag/talk.hpp"

namespace cctag {

Level::Level( std::size_t width, std::size_t height, int debug_info_level, bool cuda_allocates )
    : _debug_info_level( debug_info_level )
    , _cuda_allocates( cuda_allocates )
{
    if( _cuda_allocates ) {
    } else {
        // Allocation
        _src = cv::Mat(height, width, CV_8UC1);
        _dx =  cv::Mat(height, width, CV_16SC1 );
        _dy = cv::Mat(height, width, CV_16SC1 );
        _mag = cv::Mat(height, width, CV_16SC1 );
        _edges = cv::Mat(height, width, CV_8UC1);
    }
    _temp = cv::Mat(height, width, CV_8UC1);
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
    _edgesNotThin = cv::Mat(height, width, CV_8UC1);
#endif
}

void Level::setLevel( const cv::Mat & src, const double thrLowCanny, const double thrHighCanny, const cctag::Parameters* params )
{
  DO_TALK( std::cerr << "Enter " << __FUNCTION__ << std::endl; )
  cv::resize(src, _src, cv::Size(_src.cols,_src.rows));
  // ASSERT TODO : check that the data are allocated here
  // Compute derivative and canny edge extraction.
  cvRecodedCanny(_src,_edges,_dx,_dy, thrLowCanny * 256, thrHighCanny * 256, 3 | CV_CANNY_L2_GRADIENT, _debug_info_level, params );
  // Perform the thinning.

#ifdef CCTAG_EXTRA_LAYER_DEBUG
  _edgesNotThin = _edges.clone();
#endif
  
  thin(_edges,_temp);
  DO_TALK( std::cerr << "Leave " << __FUNCTION__ << std::endl; )
}

const cv::Mat & Level::getSrc() const
{
  return _src;
}

#ifdef CCTAG_EXTRA_LAYER_DEBUG
const cv::Mat & Level::getCannyNotThin() const
{
  return _edgesNotThin;
}
#endif

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

}
