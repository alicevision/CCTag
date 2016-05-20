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

class EdgePoint : public Vector3s
{
public:
  EdgePoint()
    : Vector3s(0, 0, 1)
    , _grad(0.f,0.f)
    , _normGrad( 0.f )
    , _before( -1 )
    , _after( -1 )
    , _votersBegin(-1)
    , _votersEnd(-1)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
    , _processedIn( false )
  {}
  
  EdgePoint( const EdgePoint& p )
    : Vector3s(p)
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _before( p._before )
    , _after( p._after )
    , _votersBegin(p._votersBegin)
    , _votersEnd(p._votersEnd)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
    , _processedIn( false )
  {}

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : Vector3s( vx, vy, 1 )
    , _grad(vdx, vdy)
    , _normGrad(std::sqrt( vdx * vdx + vdy * vdy ))
    , _before( -1 )
    , _after( -1 )
    , _votersBegin(-1)
    , _votersEnd(-1)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
    , _processedIn( false )
  {
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

  // XXX: optimize layout
private:
  Eigen::Vector2f _grad;
  float _normGrad;
public:
  int _before;
  int _after;
  int _votersBegin;
  int _votersEnd;
  uint64_t _processed;   // bitfield; must be 64-bit
  int _isMax;
  int _nSegmentOut;     // std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
  bool _processedIn;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

