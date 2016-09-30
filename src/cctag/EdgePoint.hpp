/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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

using Vector3s = Eigen::Matrix<short, 3, 1>;

class EdgePoint : public Vector3s
{
public:
  EdgePoint() = default;

  // XXX: should delete copy ctor; a lot of state is in EdgePointCollection
  // That class checks incoming pointers though.
  EdgePoint( const EdgePoint& p )
    : Vector3s(p)
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _flowLength (0)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
  {}

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : Vector3s( vx, vy, 1 )
    , _grad(vdx, vdy)
    , _normGrad(std::sqrt( vdx * vdx + vdy * vdy ))
    , _flowLength (0)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
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

private:
  Eigen::Vector2f _grad;
  float _normGrad;
public:
  float _flowLength;
  uint64_t _processed;   // bitfield; must be 64-bit
  int _isMax;
  int _nSegmentOut;     // std::size_t _nSegmentOut;
};

// Calculation: sizeof(Vector3s)==8 (3*2=6 + 2 bytes of padding to 8 bytes)
// 4*sizeof(float) == 16; plus uint64_t + 2 ints
static_assert(sizeof(EdgePoint) == 8+16+16, "EdgePoint not packed");

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

