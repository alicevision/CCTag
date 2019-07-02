/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>

#include "geom_matrix.h"

namespace cctag {

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
    cctag::geometry::matrix3x3 mHomography;
    cctag::geometry::matrix3x3 mInvHomography;
};

struct NearbyPointGrid
{
    NearbyPoint grid[5][5];

    __host__ __device__
    inline NearbyPoint& getGrid( int x, int y ) {
        if( x >= 0 && x < 5 && y >= 0 && y < 5 ) {
            return grid[y][x];
        } else {
            printf("Nearby point grid access out of bounds (%d,%d)\n", x, y );
            return grid[0][0];
        }
    }
    __host__ __device__
    inline const NearbyPoint& getGrid( int x, int y ) const {
        if( x >= 0 && x < 5 && y >= 0 && y < 5 ) {
            return grid[y][x];
        } else {
            printf("Nearby point grid access out of bounds (%d,%d)\n", x, y );
            return grid[0][0];
        }
    }
};

}; // namespace cctag

