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
#if 0
  EdgePoint()
    : Vector3s(0, 0, 1)
    , _grad(0.f,0.f)
    , _normGrad( 0.f )
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
    , _processedIn( false )
  {}
#else
  EdgePoint() = default;
#endif
    
  EdgePoint( const EdgePoint& p )
    : Vector3s(p)
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
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

