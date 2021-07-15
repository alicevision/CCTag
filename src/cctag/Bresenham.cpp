/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/EdgePoint.hpp>
#include <cctag/Bresenham.hpp>
#include <cctag/utils/FileDebug.hpp>

#include <boost/math/special_functions/sign.hpp>

#include <cmath>

namespace cctag {

static void updateXY(const float & dx, const float & dy, int & x, int & y,  float & e, int & stpX, int & stpY)
{
	float a = std::abs(dy/dx);
	stpX = boost::math::sign<int>(dx);
	stpY = boost::math::sign<int>(dy);
	e += a;
	x += stpX;
	if (e>=0.5f)
	{
		y += stpY;
		e -= 1;
	}
	return;
}

EdgePoint* gradientDirectionDescent(
        const EdgePointCollection& canny,
        const EdgePoint& p,
        int dir,
        std::size_t nmax,
        const cv::Mat & imgDx, 
        const cv::Mat & imgDy, 
        int thrGradient)
{
    EdgePoint* ret = nullptr;
    float e        = 0.0f;
    float dx       = dir * imgDx.at<short>(p.y(),p.x());
    float dy       = dir * imgDy.at<short>(p.y(),p.x());

    float dx2 = 0;
    float dy2 = 0;

    float dxRef = dx;
    float dyRef = dy;

    float adx = std::abs( dx );
    float ady = std::abs( dy );

    std::size_t n = 0;

    int stpX = 0;
    int stpY = 0;

    int x = p.x();
    int y = p.y();
    
    CCTagFileDebug::instance().newVote(x,y,dx,dy);

    if( ady > adx )
    {

        updateXY(dy,dx,y,x,e,stpY,stpX);
        CCTagFileDebug::instance().addFieldLinePoint(x, y);
        
        n = n+1;

        if ( dx*dx+dy*dy > thrGradient )
        {
            dx2 = imgDx.at<short>(p.y(),p.x());
            dy2 = imgDy.at<short>(p.y(),p.x());
            dir = boost::math::sign<float>( dx2*dxRef+dy2*dyRef );
            dx = dir*dx2;
            dy = dir*dy2;
        }

        updateXY(dy,dx,y,x,e,stpY,stpX);
        CCTagFileDebug::instance().addFieldLinePoint(x, y);
        n = n+1;

        if( x >= 0 && x < canny.shape()[0] &&
            y >= 0 && y < canny.shape()[1] )
        {
            ret = canny(x,y);
            if( ret )
            {
                    return ret;
            }
        }
        else
        {
                return nullptr;
        }

        while( n <= nmax)
        {
            updateXY(dy,dx,y,x,e, stpY,stpX);
            CCTagFileDebug::instance().addFieldLinePoint(x, y);
            n = n+1;

            if( x >= 0 && x < canny.shape()[0] &&
                y >= 0 && y < canny.shape()[1] )
            {
                ret = canny(x,y);
                if( ret )
                {
                    return ret;
                }
                else
                {
                    if( x >= 0 && x < canny.shape()[0] &&
                        ( y - stpY ) >= 0 && ( y - stpY ) < canny.shape()[1] )
                    {
                        ret = canny(x,y - stpY);              //#
                        if( ret )
                        {
                                return ret;
                        }
                    }
                    else
                    {
                            return nullptr;
                    }
                }
            }
            else
            {
                    return nullptr;
            }
        }
    }
    else
    {
        updateXY(dx,dy,x,y,e,stpX,stpY);
        CCTagFileDebug::instance().addFieldLinePoint(x, y);
        n = n+1;

        if ( dx*dx+dy*dy > thrGradient )
        {
            dx2 = imgDx.at<short>(p.y(),p.x());
            dy2 = imgDy.at<short>(p.y(),p.x());
            dir = boost::math::sign<float>( dx2*dxRef+dy2*dyRef );
            dx = dir*dx2;
            dy = dir*dy2;
        }

        updateXY(dx,dy,x,y,e,stpX,stpY);
        CCTagFileDebug::instance().addFieldLinePoint(x, y);
        n = n+1;

        if( x >= 0 && x < canny.shape()[0] &&
            y >= 0 && y < canny.shape()[1] )
        {
            ret = canny(x,y);
            if( ret )
            {
                return ret;
            }
        }
        else
        {
            return nullptr;
        }

        while( n <= nmax)
        {
            updateXY(dx,dy,x,y,e,stpX,stpY);
            CCTagFileDebug::instance().addFieldLinePoint(x, y);
            n = n+1;

            if( x >= 0 && x < canny.shape()[0] &&
                y >= 0 && y < canny.shape()[1] )
            {
                ret = canny(x,y);
                if( ret )
                {
                    return ret;
                }
                else
                {
                    if( ( x - stpX ) >= 0 && ( x - stpX ) < canny.shape()[0] &&
                        y >= 0 && y < canny.shape()[1] )
                    {
                        ret = canny(x - stpX,y);
                        if( ret )
                        {
                                return ret;
                        }
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
            else
            {
                return nullptr;
            }
        }
    }
    return nullptr;
}

} // namespace cctag
