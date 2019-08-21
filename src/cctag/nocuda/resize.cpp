/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include "cctag/nocuda/resize.hpp"

namespace cctag {

template<int SHIFT>
static void resize2( const Plane<uint8_t>& src, Plane<uint8_t>& dst )
{
#if 1
    for( int y=0; y<dst.getRows(); y++ )
    {
        int yko = std::max( (y << SHIFT) - 1, 0 );

        for( int x=0; x<dst.getCols(); x++ )
        {
            int    xko = std::max( (x << SHIFT) - 1, 0 );

            const uint8_t& val00 = src.at( xko  , yko   );
            const uint8_t& val01 = src.at( xko+1, yko   );
            const uint8_t& val10 = src.at( xko  , yko+1 );
            const uint8_t& val11 = src.at( xko+1, yko+1 );
            dst.at(x,y) = ( val00 + val10 + val01 + val11 ) >> 2;
        }
    }
#else
    for( int y=0; y<dst.getRows(); y++ )
    {
        int yko = std::max( (y << SHIFT) - 1, 0 );
        for( int x=0; x<dst.getCols(); x++ )
        {
            int    xko = std::max( (x << SHIFT) - 1, 0 );
            uint8_t val = (uint8_t)( src.at( xko, yko ) );
            dst.at(x,y) = val;
        }
    }
#endif
}

void resize( const Plane<uint8_t>& src, Plane<uint8_t>& dst )
{
    if( src.getRows() == 1 * dst.getRows() &&
        src.getCols() == 1 * dst.getCols() )
    {
        std::cerr << "ratio 1" << std::endl;
        memcpy( dst.getBuffer(), src.getBuffer(), dst.getRows()*dst.getCols() );
        return;
    }

    if( src.getRows() == 2 * dst.getRows() && src.getCols() == 2 * dst.getCols() ) {
        std::cerr << "ratio 2" << std::endl;
        resize2<1>( src, dst );
        return;
    }
    if( src.getRows() == 4 * dst.getRows() && src.getCols() == 4 * dst.getCols() ) {
        std::cerr << "ratio 4" << std::endl;
        resize2<2>( src, dst );
        return;
    }
    if( src.getRows() == 8 * dst.getRows() && src.getCols() == 8 * dst.getCols() ) {
        std::cerr << "ratio 8" << std::endl;
        resize2<3>( src, dst );
        return;
    }

    const float row_ratio = (float)src.getRows() / (float)dst.getRows();
    const float col_ratio = (float)src.getCols() / (float)dst.getCols();
    if( row_ratio >= 1.0f )
    {
        std::cerr << "ratio, row=" << row_ratio << " col=" << col_ratio << std::endl;

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
