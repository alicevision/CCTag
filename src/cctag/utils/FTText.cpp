/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cctag/utils/FTText.hpp"

namespace cctag
{

#ifdef CCTAG_USE_FREETYPE
bool FT2::_failed = false;
FT2* FT2::_opened = 0;

FT2::~FT2( )
{
    FT_Done_Face( _face );
    FT_Done_FreeType( _lib );
    _failed = false;
}

void FT2::init()
{
    FT_Error   error;
    FT_Library lib;
    FT_Face    face;

    if( _failed ) return;

    if( _opened == 0 )
    {
        error = FT_Init_FreeType( &lib );
        if( error )
        {
            _failed = true;
            return;
        }

        error = FT_New_Face( lib,
                             "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                             0,
                             &face );
        if( error == FT_Err_Unknown_File_Format )
        {
            _failed = true;
            return;
        }
        else if( error )
        {
            _failed = true;
            return;
        }

        _opened = new FT2( lib, face );
    }
}

void FT2::uninit()
{
    if( _failed ) return;

    delete _opened;
    _opened = 0;
}
#endif // CCTAG_USE_FREETYPE

// from here: https://www.freetype.org/freetype2/docs/tutorial/example1.c

void FT2::write_text( Plane<Color>& plane, int x, int y, const char* str )
{
#ifdef CCTAG_USE_FREETYPE
    init();

    /* use 50pt at 100dpi */
    FT_Error error = FT_Set_Char_Size( _opened->_face, 50 * 64, 0, 100, 0 ); /* set character size */
    /* error handling omitted */

    FT_GlyphSlot  slot;
    slot = _opened->_face->glyph;

    // FT_Vector pen;
    // pen.x = x;
    // pen.y = y;
    int len = strlen(str);
    for( int i=0; i<len; i++ )
    {
        error = FT_Load_Char( _opened->_face, str[i], FT_LOAD_RENDER );
        if( error ) continue;

        FT_GlyphSlot slot   = _opened->_face->glyph;
        if( slot->format != FT_GLYPH_FORMAT_BITMAP )
        {
            error = FT_Render_Glyph( slot, FT_RENDER_MODE_NORMAL );
        }

        for( int h=0; h<slot->bitmap.rows; h++ )
        {
            for( int w=0; w<slot->bitmap.width; w++ )
            {
                unsigned char val = slot->bitmap.buffer[ h * slot->bitmap.pitch + w ];
                if( val > 128 )
                {
                    if( x+w >= 1.0f && x+w < plane.getCols()-1.0f &&
                        y+h >= 1.0f && y+h < plane.getRows()-1.0f )
                    {
                        plane.at( (int)roundf(x+w), (int)roundf(y+h) ) = color_red;
                    }
                    
                }
            }
        }

        /* increment pen position */
        x += slot->bitmap.width + 1;
        // x += slot->advance.x;
        // y += slot->advance.y;
    }
#else
    static bool said = false;
    if( !said )
    {
        std::cerr << "CCTag built without freetype - no text in visual debug output" << std::endl;
        said = true;
    }
#endif // CCTAG_USE_FREETYPE
}

}; // namespace cctag;

