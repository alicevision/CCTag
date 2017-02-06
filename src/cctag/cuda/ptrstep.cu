/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "ptrstep.h"

namespace cctag {

PtrStepSzbClone::PtrStepSzbClone( const cv::cuda::PtrStepSzb& orig )
    : e ( orig )
{
    e.data = new uint8_t[ orig.rows * orig.step ];
    memcpy( e.data, orig.data, orig.rows * orig.step );
}

PtrStepSzbClone::~PtrStepSzbClone( )
{
    delete [] e.data;
}

PtrStepSzbNull::PtrStepSzbNull( const int width, const int height )
{
    e.step = width;
    e.cols = width;
    e.rows = height;
    e.data = new uint8_t[ e.rows * e.step ];
    memset( e.data, 0, e.rows * e.step );
}

PtrStepSzbNull::~PtrStepSzbNull( )
{
    delete [] e.data;
}

} // namespace cctag

