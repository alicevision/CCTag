/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef __CUDACC__
 //
 // When compiling .cu files, there's a bunch of stuff that doesn't work with msvc:
 //
#if defined(_MSC_VER)
#  define BOOST_NO_CXX14_DIGIT_SEPARATORS
#  define BOOST_NO_CXX11_UNICODE_LITERALS
#  define BOOST_PP_VARIADICS 0
#endif
#  define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#endif

#include <cuda_runtime.h>


 // https://svn.boost.org/trac/boost/ticket/11897
 // This is fixed in 7.5. As the following version macro was introduced in 7.5 an existance
 // check is enough to detect versions < 7.5
 #if BOOST_CUDA_VERSION < 7050000
 #   define BOOST_NO_CXX11_VARIADIC_TEMPLATES
 #endif
 // The same bug is back again in 8.0:
 #if (BOOST_CUDA_VERSION > 8000000) && (BOOST_CUDA_VERSION < 8010000)
 #   define BOOST_NO_CXX11_VARIADIC_TEMPLATES
 #endif
 // Most recent CUDA (8.0) has no constexpr support in msvc mode:
 #if defined(_MSC_VER)
 #  define BOOST_NO_CXX11_CONSTEXPR
 #endif

#ifdef __CUDACC__
// And this one effects the NVCC front end,
// See https://svn.boost.org/trac/boost/ticket/13049
//
#if (BOOST_CUDA_VERSION >= 8000000) && (BOOST_CUDA_VERSION < 8010000)
#  define BOOST_NO_CXX11_NOEXCEPT
#endif

#endif


