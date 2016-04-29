#ifndef VISION_EDGEPOINT_HPP_
#define VISION_EDGEPOINT_HPP_

#include <cctag/geometry/Point.hpp>

#include <cstddef>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>

namespace cctag
{

class Label;

class EdgePoint : public cctag::Point2d<Eigen::Vector3i>
{
public:
  EdgePoint()
    : Point2d(0, 0)
    , _normGrad( -1.0 )
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
  {
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
    _grad << vdx , vdy;
  }

  void init( const int vx, const int vy, const float vdx, const float vdy )
  {
    x() = vx;
    y() = vy;
    w() = 1;
    _grad << vdx , vdy;
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
  }

  cctag::Point2d<Eigen::Vector2f> gradient() const
  {
    return _grad ;
  }

  double normGradient() const
  {
    return _normGrad ;
  }

  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

  std::vector<EdgePoint*> _voters;
  EdgePoint* _before;
  EdgePoint* _after;
  size_t _processed;
  bool _processedIn;
  ssize_t _isMax;
  ssize_t _edgeLinked;
  ssize_t _nSegmentOut; // std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
  
private:
  Eigen::Vector2f _grad;
  double _normGrad;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

