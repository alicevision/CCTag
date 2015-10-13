#ifndef _CCTAG_LINE_HPP_
#define	_CCTAG_LINE_HPP_

#include <cctag/geometry/point.hpp>
#include <cctag/params.hpp>

#include <boost/numeric/ublas/vector.hpp>

namespace cctag {

class ImageCut
{
  
public:
  ImageCut()
  {
    _outOfBounds = false;
  }
 
  ImageCut( Point2dN<double> pStart, DirectedPoint2d<double> pStop )
  : _start(pStart), _stop(pStop)
  {
    _imgSignal.resize(cctag::kDefaultSampleCutLength);
    _outOfBounds = false;
  }
  
    ImageCut( Point2dN<double> pStart, DirectedPoint2d<double> pStop, const std::size_t nSamples)
  : _start(pStart), _stop(pStop)
  {
    _imgSignal.resize(nSamples);
    _outOfBounds = false;
  }
  
  virtual ~ImageCut()
  {
  }
  
  Point2dN<double> _start;
  DirectedPoint2d<double> _stop;
  boost::numeric::ublas::vector<double> _imgSignal; //< image signal
  bool _outOfBounds;
  
};

}


#endif
