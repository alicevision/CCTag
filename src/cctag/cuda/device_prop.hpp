/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <vector>
#include <cctag/cuda/cctag_cuda_runtime.h>

namespace cctag {

class device_prop_t
{
    int _num_devices;
    std::vector<cudaDeviceProp*> _properties;
public:
    explicit device_prop_t( bool output = false );
    ~device_prop_t( );

    void print( );
    void set( int n );
};

}; // namespace cctag

