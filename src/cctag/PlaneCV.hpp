/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_PLANECV_HPP_
#define _CCTAG_PLANECV_HPP_

#include <cstring>

#include "cctag/Plane.hpp"

#include <opencv2/opencv.hpp>

namespace cctag {

/*************************************************************
 * PlaneType
 *************************************************************/

template<typename SubType> struct PlaneType
{
    int cvType() { return CV_8UC1; }
};

template<> struct PlaneType<uint8_t>
{
    int cvType() { return CV_8UC1; }
};

template<> struct PlaneType<int16_t>
{
    int cvType() { return CV_16SC1; }
};

template<> struct PlaneType<int32_t>
{
    int cvType() { return CV_32SC1; }
};

/*************************************************************
 * function Plane
 *************************************************************/

template<typename Type>
cv::Mat planeToMat( cctag::Plane<Type>& plane )
{
    PlaneType<Type> t;
    cv::Mat mat( plane.getRows(), plane.getCols(),
                 t.cvType(),
                 (void*)plane.getBuffer() );
    return mat;
}

template<typename Type>
cv::Mat planeToMat( const cctag::Plane<Type>& plane )
{
    PlaneType<Type> t;
    cv::Mat mat( plane.getRows(), plane.getCols(),
                 t.cvType(),
                 (void*)plane.getBuffer() );
    return mat;
}

} // namespace cctag

#endif

