#ifndef _POPART_VISION_EDGEPOINT_HPP_
#define _POPART_VISION_EDGEPOINT_HPP_

#include <cctag/geometry/point.hpp>

#include <cstddef>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>

namespace popart
{
namespace vision
{

class Label;

class EdgePoint : public popart::Point2dN<int>
{
public:
  EdgePoint()
    : popart::Point2dN<int>( 0, 0 )
    , _normGrad( -1.0 )
    , _before( NULL )
    , _after( NULL )
    , _processed( -1 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    ,_processedAux(false)

  {}

  EdgePoint( const EdgePoint& p )
    : popart::Point2dN<int>( p )
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _before( p._before )
    , _after( p._after )
    , _processed( -1 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
  {}

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : popart::Point2dN<int>( vx, vy )
    , _before( NULL )
    , _after( NULL )
    , _processed( -1 )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
  {
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
    _grad = popart::Point2dN<double>( (double) vdx , (double) vdy );
  }

  virtual ~EdgePoint() {}

  inline popart::Point2dN<double> gradient() const
  {
    return _grad ;
  }

  inline double normGradient() const
  {
    return _normGrad ;
  }

  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

  popart::Point2dN<double> _grad;
  double _normGrad;
  EdgePoint* _before;
  EdgePoint* _after;
  ssize_t _processed;
  bool _processedIn;
  ssize_t _isMax;
  ssize_t _edgeLinked;
  std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

}
}

#endif

