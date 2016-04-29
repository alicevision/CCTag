#ifndef VISION_EDGEPOINT_HPP_
#define VISION_EDGEPOINT_HPP_

#include <cctag/geometry/Point.hpp>

#include <cstddef>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>
#include <stdio.h>

namespace cctag
{

class Label;

class EdgePoint : public cctag::Point2dN<int>
{
public:
  EdgePoint()
    : cctag::Point2dN<int>( 0, 0 )
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
    : cctag::Point2dN<int>( p )
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
    : cctag::Point2dN<int>( vx, vy )
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
    _grad = cctag::Point2dN<double>( (double) vdx , (double) vdy );
  }

  void init( const int vx, const int vy, const float vdx, const float vdy )
  {
    this->setX( vx );
    this->setY( vy );
    this->setW( 1 );
    _grad = cctag::Point2dN<double>( (double) vdx , (double) vdy );
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
  }

  virtual ~EdgePoint() {}

  inline cctag::Point2dN<double> gradient() const
  {
    return _grad ;
  }

  inline double normGradient() const
  {
    return _normGrad ;
  }

  inline void print( FILE* fileHandle ) const {
    fprintf( fileHandle, "(%d,%d) (%3.3g,%3.3g) %3.3g %ld %ld %ld %f %d\n", x(), y(), _grad.x(), _grad.y(), _normGrad, _isMax, _edgeLinked, _nSegmentOut, _flowLength, _processedAux );
  }

  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

  cctag::Point2dN<double> _grad;
  double _normGrad;
  EdgePoint* _before;
  EdgePoint* _after;
  ssize_t _processed;
  bool _processedIn;
  ssize_t _isMax;
  ssize_t _edgeLinked;
  ssize_t _nSegmentOut; // std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

