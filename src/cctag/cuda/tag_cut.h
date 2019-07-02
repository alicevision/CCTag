/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>

namespace cctag {

namespace identification {

struct CutStruct
{
    float2 start;     // this is moving: initially set to the approximate center of the ellipse
    float2 stop;      // not moving inside cctag::refineConicFamilyGlob
    float  beginSig;  // never changes (for all tags!)
    float  endSig;    // never changes (for all tags!)
    int    sigSize;   // never changes (for all tags!)
};

struct CutSignals
{
    uint32_t outOfBounds;
    float    sig[127];
};

} // namespace identification

struct CutStructGrid
{
    identification::CutStruct grid[22];

    __host__ __device__
    inline identification::CutStruct& getGrid( int cut ) {
        if( cut >= 0 && cut < 22 ) {
            return grid[cut];
        } else {
            printf("Cut struct grid access out of bounds (%d)\n", cut );
            return grid[0];
        }
    }
    __host__ __device__
    inline const identification::CutStruct& getGrid( int cut ) const {
        if( cut >= 0 && cut < 22 ) {
            return grid[cut];
        } else {
            printf("Cut struct grid access out of bounds (%d)\n", cut );
            return grid[0];
        }
    }
};

struct CutSignalGrid
{
    identification::CutSignals grid[22][5][5];

    __host__ __device__
    inline identification::CutSignals& getGrid( int cut, int x, int y ) {
        if( cut >= 0 && cut < 22 && x >= 0 && x < 5 && y >= 0 && y < 5 ) {
            return grid[cut][y][x];
        } else {
            printf("Cut signal grid access out of bounds (%d,%d,%d)\n", cut, x, y );
            return grid[0][0][0];
        }
    }
    __host__ __device__
    inline const identification::CutSignals& getGrid( int cut, int x, int y ) const {
        if( cut >= 0 && cut < 22 && x >= 0 && x < 5 && y >= 0 && y < 5 ) {
            return grid[cut][y][x];
        } else {
            printf("Cut signal grid access out of bounds (%d,%d,%d)\n", cut, x, y );
            return grid[0][0][0];
        }
    }
};

}; // namespace cctag

