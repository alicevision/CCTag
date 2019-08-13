/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include "cctag/PlaneCV.hpp"

namespace cctag {

void resize( const Plane<uint8_t>& src, Plane<uint8_t>& dst )
{
    if( src.getRows() == 1 * dst.getRows() &&
        src.getCols() == 1 * dst.getCols() )
    {
        memcpy( dst.getBuffer(), src.getBuffer(), dst.getRows()*dst.getCols() );
        return;
    }

    if( src.getRows() == 2 * dst.getRows() &&
        src.getCols() == 2 * dst.getCols() )
    {
        for( int y=0; y<dst.getRows(); y++ )
        {
            const float yko = y << 1;
            for( int x=0; x<dst.getCols(); x++ )
            {
                const float    xko = x << 1;
                const uint8_t& val00 = src.at( xko+0, yko+0 );
                const uint8_t& val01 = src.at( xko+1, yko+0 );
                const uint8_t& val10 = src.at( xko+0, yko+1 );
                const uint8_t& val11 = src.at( xko+1, yko+1 );
                dst.at(x,y) = ( val00 + val10 + val01 + val11 ) >> 2;
            }
        }
        return;
    }

    const float row_ratio = (float)src.getRows() / (float)dst.getRows();
    const float col_ratio = (float)src.getCols() / (float)dst.getCols();
    if( row_ratio >= 1.0f )
    {
        /* destination is smaller, just pick source pixels */
        for( int y=0; y<dst.getRows(); y++ )
        {
            const float yko = y*row_ratio;
            const int ybase = yko - int(yko) < 0.5f ? yko : yko + 1;
            for( int x=0; x<dst.getCols(); x++ )
            {
                const float xko = x*col_ratio;
                const int xbase = xko - int(xko) < 0.5f ? xko : xko + 1;
                dst.at(x,y) = src.at( xbase, ybase );
            }
        }
    }
    else
    {
        std::cerr << "ratio, row=" << row_ratio << " col=" << col_ratio << std::endl;

        for( int y=0; y<dst.getRows(); y++ )
        {
            const float yko = y*row_ratio;
            const int ybase = (int)yko;
            const float yw = yko - ybase;
            const float ycw = 1.0f - yw;

            for( int x=0; x<dst.getCols(); x++ )
            {
                const float xko = x*col_ratio;
                const int xbase = (int)xko;
                const float xw = xko - xbase;
                const float xcw = 1.0f - xw;

                const float val00 = xcw * ycw * src.at( xbase+0, ybase+0 );
                const float val01 = xw  * ycw * src.at( xbase+1, ybase+0 );
                const float val10 = xcw * yw  * src.at( xbase+0, ybase+1 );
                const float val11 = xw  * yw  * src.at( xbase+1, ybase+1 );
                dst.at(x,y) = uint8_t( val00 + val01 + val10 + val11 );
            }
        }
    }
}

}
