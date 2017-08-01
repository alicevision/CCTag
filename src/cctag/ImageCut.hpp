/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_LINE_HPP_
#define	_CCTAG_LINE_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/Params.hpp>

namespace cctag {

class ImageCut
{
  
public:
  ImageCut()
  {
    _outOfBounds = false;
  }
 
  ImageCut( Point2d<Eigen::Vector3f> pStart, DirectedPoint2d<Eigen::Vector3f> pStop )
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(0.f), _endSig(1.f)
  {
    _imgSignal.resize(cctag::kDefaultSampleCutLength);
  }
  
  ImageCut( Point2d<Eigen::Vector3f> pStart, DirectedPoint2d<Eigen::Vector3f> pStop, const float start, const float stop)
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(start), _endSig(stop)
  {
    _imgSignal.resize(cctag::kDefaultSampleCutLength);
    _outOfBounds = false;
  }
  
  ImageCut( Point2d<Eigen::Vector3f> pStart, DirectedPoint2d<Eigen::Vector3f> pStop, const std::size_t nSamples)
  : _start(pStart), _stop(pStop), _beginSig(0.f), _endSig(1.f), _outOfBounds(false)
  {
    _imgSignal.resize(nSamples);
  }
    
  ImageCut( Point2d<Eigen::Vector3f> pStart, DirectedPoint2d<Eigen::Vector3f> pStop, const float start, const float stop, const std::size_t nSamples)
  : _start(pStart), _stop(pStop), _outOfBounds(false), _beginSig(start), _endSig(stop)
  {
    _imgSignal.resize(nSamples);
    _outOfBounds = false;
  }
  
  const Point2d<Eigen::Vector3f> & start() const { return _start; }
  Point2d<Eigen::Vector3f> & start() { return _start; }
  
  const DirectedPoint2d<Eigen::Vector3f> & stop() const { return _stop; }
  DirectedPoint2d<Eigen::Vector3f> & stop() { return _stop; }
  
  const std::vector<float> & imgSignal() const { return _imgSignal; }
  std::vector<float> & imgSignal() { return _imgSignal; }
  
  float beginSig() const { return _beginSig; }
  
  float endSig() const { return _endSig; }
  
  bool outOfBounds() const { return _outOfBounds; }
  
  void setOutOfBounds(const bool outOfBounds) { _outOfBounds = outOfBounds; }
  
  virtual ~ImageCut() = default;
  
private:
  
  /** Start point of the image cut
   * @todo This value is invalid after the optimization (neither CPU or GPU)
   */
  Point2d<Eigen::Vector3f> _start;
  
  // Stop point of the image cut
  DirectedPoint2d<Eigen::Vector3f> _stop;
  
  /** 1D rectified image signal along the segment [_start,_stop]
   * @note For the GPU case, this information never exists on the CPU
   */
  std::vector<float> _imgSignal; //< image signal
  
  // False by default. This boolean reveals if any of the points lying on the segment
  // [_start, _stop] are outside of the image
  bool _outOfBounds;
  
  // Scalar value in [0,1] representing from where, along the segment [_start,_stop] 
  // whose extremities corresponds to 0 and 1 resp., the image signal in stored in 
  // _imgSignal 
  float _beginSig;
  
  // Scalar value in [_beginSig,1] representing to where, along the segment [_start,_stop] 
  // whose extremities corresponds to 0 and 1 resp., the image signal in stored in 
  // _imgSignal
  float _endSig;
  
};

}


#endif
