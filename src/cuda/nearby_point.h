/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>

#include "geom_matrix.h"

namespace popart {

struct NearbyPoint
{
    float2 point;
    float  result;
    int    resSize;
    bool   readable;
    float  residual;

    /* These homographies are computed once for each NearbyPoint,
     * and used for all of its Cuts. The best one must be returned.
     */
    popart::geometry::matrix3x3 mHomography;
    popart::geometry::matrix3x3 mInvHomography;
};

}; // namespace popart

