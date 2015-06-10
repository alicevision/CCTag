#ifndef _CCTAG_LINE_HPP_
#define	_CCTAG_LINE_HPP_

#include <cctag/geometry/point.hpp>
#include <cctag/params.hpp>

#include <boost/numeric/ublas/vector.hpp>

namespace cctag {

class ImageCut
{
  
  ImageCut()
  {
    
  }
 
  ImageCut(Point2dN<double> pStart, Point2dN<double> pStop)
  : _start(pStart), _stop(pStop)
  {
    _imgSignal.reserve(cctag::kDefaultSampleCutLength);
  }
  
  virtual ~ImageCut()
  {
  }
  
  Point2dN<double> _start;
  Point2dN<double> _stop;
  boost::numeric::ublas::vector<double> _imgSignal; //< image signal
  
};

}


#endif
