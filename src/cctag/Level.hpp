/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_LEVEL_HPP
#define	_CCTAG_LEVEL_HPP

#include <opencv2/opencv.hpp>
#include "cctag/Plane.hpp"

namespace cctag {
    class TagPipe;
};

namespace cctag {

struct Parameters;

class Level
{
public:
  
  Level( std::size_t width, std::size_t height, int debug_info_level, bool cuda_allocates = false );
  
  ~Level( );

  void setLevel( const Plane<uint8_t>& src,
                 float thrLowCanny,
                 float thrHighCanny,
                 const cctag::Parameters* params );
#ifdef CCTAG_WITH_CUDA
  void setLevel( cctag::TagPipe* cuda_pipe,
                 const cctag::Parameters& params );
#endif // CCTAG_WITH_CUDA

  Plane<uint8_t>&       getSrc();
  const Plane<int16_t>& getDx() const;
  const Plane<int16_t>& getDy() const;
  const Plane<int16_t>& getMag() const; 
  const Plane<uint8_t>& getEdges() const;
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
  const Plane<uint8_t>& getCannyNotThin() const;
#endif
  
  inline std::size_t width() const
  {
    return _cols;
  }
  
  inline std::size_t height() const
  {
    return _rows;
  }
  

private:
  int         _level;
  bool        _cuda_allocates;
  bool        _mat_initialized_from_cuda;
  std::size_t _cols;
  std::size_t _rows;
  
  Plane<int16_t>* _dx;
  Plane<int16_t>* _dy;
  Plane<int16_t>* _mag;
  Plane<uint8_t>* _src;
  Plane<uint8_t>* _edges;
  Plane<uint8_t>  _temp;
  
#ifdef CCTAG_EXTRA_LAYER_DEBUG
  Plane<uint8_t> _edgesNotThin;
#endif
};

}

#endif	/* _CCTAG_LEVEL_HPP */
