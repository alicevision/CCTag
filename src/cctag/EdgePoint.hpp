#ifndef VISION_EDGEPOINT_HPP_
#define VISION_EDGEPOINT_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/utils/Defines.hpp>

#include <cstddef>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>

namespace cctag
{

class Label;

using Vector3s = Eigen::Matrix<short, 3, 1>;

class EdgePoint : public cctag::Point2d<Vector3s>
{
public:
#ifdef WITH_CUDA
  EdgePoint() = default;  // don't want to do any work in this in cuda part where it's constructed
#else
  EdgePoint()
    : Point2d(0, 0)
    , _grad(0.f,0.f)
    , _normGrad( 0.f )
    , _before( NULL )
    , _after( NULL )
    , _processed( 0 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    ,_processedAux(false)
  {}
#endif
  
  EdgePoint( const EdgePoint& p )
    : Point2d( p )
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _before( p._before )
    , _after( p._after )
    , _processed( 0 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
  {}

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : Point2d( vx, vy )
    , _before( NULL )
    , _after( NULL )
    , _processed( 0 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
    , _grad(vdx, vdy)
    , _normGrad(0.f)
  {
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
  }

  Eigen::Vector2f gradient() const
  {
    return _grad;
  }
  
  float dX() const
  {
    return _grad(0);
  }
  
   float dY() const
  {
    return _grad(1);
  }

  float normGradient() const
  {
    return _normGrad ;
  }

  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

  EdgePoint* _before;
  EdgePoint* _after;
  size_t _processed;
  ssize_t _isMax;
  ssize_t _edgeLinked;
  ssize_t _nSegmentOut; // std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
  bool _processedIn;
  std::vector<EdgePoint*> _voters;
  
private:
  Eigen::Vector2f _grad;
  float _normGrad;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

