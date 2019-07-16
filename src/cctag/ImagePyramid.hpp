/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_IMAGEPYRAMID_HPP
#define	_CCTAG_IMAGEPYRAMID_HPP

#include "cctag/Level.hpp"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <cstddef>
#include <vector>

namespace cctag {

struct Parameters; // forward declaration

class ImagePyramid
{
public:
  ImagePyramid();
  
  ImagePyramid( std::size_t width, std::size_t height, std::size_t nLevels, bool cuda_allocates );
  
  ~ImagePyramid();

  Level* getLevel( std::size_t level ) const;
  
  std::size_t getNbLevels() const;
  
    /* The pyramid building function is never called if CUDA is used.
     */
  void build(const cv::Mat & src, float thrLowCanny, float thrHighCanny, const cctag::Parameters* params );

private:
  std::vector<Level*> _levels;
};

void sIntToUchar(const cv::Mat & src, cv::Mat & dst);

}

#endif	/* _CCTAG_IMAGEPYRAMID_HPP */

