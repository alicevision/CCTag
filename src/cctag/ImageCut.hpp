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
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(0.0), _endSig(1.0)
  {
    _imgSignal.resize(cctag::kDefaultSampleCutLength);
  }
  
  ImageCut( Point2dN<double> pStart, DirectedPoint2d<double> pStop, const double start, const double stop)
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(start), _endSig(stop)
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
    
  ImageCut( Point2dN<double> pStart, DirectedPoint2d<double> pStop, const double start, const double stop, const std::size_t nSamples)
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(start), _endSig(stop)
  {
    _imgSignal.resize(nSamples);
    _outOfBounds = false;
  }
  
  const Point2dN<double> & start() const { return _start; }
  Point2dN<double> & start() { return _start; }
  
  const DirectedPoint2d<double> & stop() const { return _stop; }
  DirectedPoint2d<double> & stop() { return _stop; }
  
  const boost::numeric::ublas::vector<double> & imgSignal() const { return _imgSignal; }
  boost::numeric::ublas::vector<double> & imgSignal() { return _imgSignal; }
  
  double beginSig() { return _beginSig; }
  
  double endSig() { return _endSig; }
  
  bool outOfBounds() const { return _outOfBounds; }
  
  void setOutOfBounds(const bool outOfBounds) { _outOfBounds = outOfBounds; }
  
  virtual ~ImageCut() {}
  
private:
  
  // Start point of the image cut
  Point2dN<double> _start;
  
  // Stop point of the image cut
  DirectedPoint2d<double> _stop;
  
  // 1D rectified image signal along the segment [_start,_stop]
  boost::numeric::ublas::vector<double> _imgSignal; //< image signal
  
  // False by default. This boolean reveals if any of the points lying on the segment
  // [_start, _stop] are outside of the image
  bool _outOfBounds;
  
  // Scalar value in [0,1] representing from where, along the segment [_start,_stop] 
  // whose extremities corresponds to 0 and 1 resp., the image signal in stored in 
  // _imgSignal 
  double _beginSig;
  
  // Scalar value in [_beginSig,1] representing to where, along the segment [_start,_stop] 
  // whose extremities corresponds to 0 and 1 resp., the image signal in stored in 
  // _imgSignal
  double _endSig;
  
};

}


#endif
