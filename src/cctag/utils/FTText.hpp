/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_CCTAG_FTTEXT_HPP_
#define _CCTAG_CCTAG_FTTEXT_HPP_


#include "cctag/Plane.hpp"

#ifdef CCTAG_USE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#endif // CCTAG_USE_FREETYPE

namespace cctag
{

class FT2
{
#ifdef CCTAG_USE_FREETYPE
    static FT2* _opened;
    static bool _failed;

public:
    FT_Library  _lib;
    FT_Face     _face;
public:
    FT2( FT_Library& lib, FT_Face& face )
        : _lib( lib )
        , _face( face )
    { }

    ~FT2();

    static void init();
    static void uninit();
#endif // CCTAG_USE_FREETYPE

    static void write_text( Plane<Color>& plane, int x, int y, const char* text );
};

}; // namespace cctag

#endif // _CCTAG_CCTAG_FTTEXT_HPP_

