#pragma once

#include <cctag/geometry/point.hpp>
#include "cuda/ptrstep.h"
#include "cctag/Level.hpp"

#include <cstddef>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>

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
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    ,_processedAux(false)
  {
    _processed = -1;
  }

  EdgePoint( const EdgePoint& p )
    : cctag::Point2dN<int>( p )
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _before( p._before )
    , _after( p._after )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
  {
    _processed = -1;
  }

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : cctag::Point2dN<int>( vx, vy )
    , _before( NULL )
    , _after( NULL )
    , _processedIn( false )
    , _isMax( -1 )
    , _edgeLinked( -1 )
    , _nSegmentOut(-1)
    , _flowLength (0)
    , _processedAux(false)
  {
    _normGrad = std::sqrt( vdx * vdx + vdy * vdy );
    _grad = cctag::Point2dN<double>( (double) vdx , (double) vdy );
    _processed = -1;
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

    inline void setProcessed( Level* level, int32_t val )
    {
        level->setProcessed( x(), y(), val );
        // _processed = val;
    }

    inline ssize_t getProcessed( Level* level ) const
    {
        return level->getProcessed( x(), y() );
        // return _processed;
    }


  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

  cctag::Point2dN<double> _grad;
  double _normGrad;
  EdgePoint* _before;
  EdgePoint* _after;
  bool _processedIn;
  ssize_t _isMax;
  ssize_t _edgeLinked;
  ssize_t _nSegmentOut; // std::size_t _nSegmentOut;
  float _flowLength;
  bool _processedAux;
private:
  ssize_t _processed;
};

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

