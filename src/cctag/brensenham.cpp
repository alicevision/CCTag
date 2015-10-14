#include <cctag/EdgePoint.hpp>
#include <cctag/brensenham.hpp>
#include <cctag/fileDebug.hpp>

#include <cctag/toolbox/gilTools.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/multi_array.hpp>
#include <boost/math/special_functions/sign.hpp>

#include <cmath>

namespace cctag {

void bresenham( const boost::gil::gray8_view_t & sView, const cctag::Point2dN<int>& p, const cctag::Point2dN<float>& dir, const std::size_t nmax )
{
	cctag::Point2dN<int> pStart = p;
	float e        = 0.0f;
	float dx       = dir.x();
	float dy       = dir.y();

	float adx = std::abs( dx );
	float ady = std::abs( dy );

	std::size_t n = 0;

	boost::gil::fill_black( sView );

	if( ady > adx )
	{
		float a   = adx / ady;                          //#
		int stp_x = boost::math::sign( dx );
		int stp_y = boost::math::sign( dy );

		int x = p.x();
		int y = p.y();
		*sView.xy_at( x, y ) = *sView.xy_at( x, y ) + 50;

		n = n + 1;
		e = e + a;
		y = y + stp_y;                                  //#

		if( e >= 0.5f )
		{
			x = x + stp_x;                              //#
			e = e - 1.0f;
		}

		n = n + 1;
		e = e + a;
		y = y + stp_y;                                  //#
		if( e >= 0.5f )
		{
			x = x + stp_x;                              //#
			e = e - 1.0f;
		}
		if( x >= 0 && x < sView.width() &&
		    y >= 0 && y < sView.height() )
		{
			*sView.xy_at( x, y ) = *sView.xy_at( x, y ) + 50;
		}
		else
		{
			return;
		}

		while( n <= nmax )
		{
			n = n + 1;
			e = e + a;
			y = y + stp_y;                              //#
			if( e >= 0.5f )
			{
				x = x + stp_x;                          //#
				e = e - 1.0f;
			}

			if( x >= 0 && x < sView.width() &&
			    y >= 0 && y < sView.height() )
			{
				*sView.xy_at( x, y ) = *sView.xy_at( x, y ) + 50;
			}
			else
			{
				break;
			}
		}
	}
	else
	{
		float a   = ady / adx;
		int stp_x = boost::math::sign( dx );
		int stp_y = boost::math::sign( dy );

		int x = p.x();
		int y = p.y();

		n = n + 1;
		e = e + a;
		x = x + stp_x;

		if( e >= 0.5f )
		{
			y = y + stp_y;
			e = e - 1.0f;
		}

		n = n + 1;
		e = e + a;
		x = x + stp_x;
		if( e >= 0.5f )
		{
			y = y + stp_y;
			e = e - 1;
		}

		if( x >= 0 && x < sView.width() &&
			y >= 0 && y < sView.height() )
		{
			*sView.xy_at( x, y ) = *sView.xy_at( x, y ) + 50;
		}
		else
		{
			return;
		}

		while( n <= nmax )
		{
			n = n + 1;
			e = e + a;
			x = x + stp_x;
			if( e >= 0.5f )
			{
				y = y + stp_y;
				e = e - 1.0f;
			}

			if( x >= 0 && x < sView.width() &&
				y >= 0 && y < sView.height() )
			{
				*sView.xy_at( x, y ) = *sView.xy_at( x, y ) + 50;
			}
			else
			{
				break;
			}
		}
	}
}

EdgePoint* bresenham( const boost::multi_array<EdgePoint*, 2> & canny, const EdgePoint& p, const int dir, const std::size_t nmax )
{
	EdgePoint* ret = NULL;
	float e        = 0.0f;
	float dx       = dir * p._grad.x();
	float dy       = dir * p._grad.y();

	float adx = std::abs( dx );
	float ady = std::abs( dy );

	std::size_t n = 0;

	if( ady > adx )
	{
		float a   = adx / ady;                          //#
		int stp_x = boost::math::sign( dx );
		int stp_y = boost::math::sign( dy );

		int x = p.x();
		int y = p.y();

		n = n + 1;
		e = e + a;
		y = y + stp_y;                                  //#

		if( e >= 0.5f )
		{
			x = x + stp_x;                              //#
			e = e - 1.0f;
		}

		// Partie commenté dû à la différence de réponse du détecteur de contour
		/*if ( x >= 0 && x < canny.shape()[0] &&
		     y >= 0 && y < canny.shape()[1] )
		   {
		    ret = canny[x][y];
		    if ( ret )
		    {
		        return ret;
		    }
		   }
		   else
		   {
		    return NULL;
		   }*/
		n = n + 1;
		e = e + a;
		y = y + stp_y;                                  //#
		if( e >= 0.5f )
		{
			x = x + stp_x;                              //#
			e = e - 1.0f;
		}
		if( x >= 0 && x < canny.shape()[0] &&
		    y >= 0 && y < canny.shape()[1] )
		{
			ret = canny[x][y];
			if( ret )
			{
				return ret;
			}
		}
		else
		{
			return NULL;
		}

		while( n <= nmax )
		{
			n = n + 1;
			e = e + a;
			y = y + stp_y;                              //#
			if( e >= 0.5f )
			{
				x = x + stp_x;                          //#
				e = e - 1.0f;
			}

			if( x >= 0 && x < canny.shape()[0] &&
			    y >= 0 && y < canny.shape()[1] )
			{
				ret = canny[x][y];
				if( ret )
				{
					return ret;
				}
				else
				{
					if( x >= 0 && x < canny.shape()[0] &&
					    ( y - stp_y ) >= 0 && ( y - stp_y ) < canny.shape()[1] )
					{
						ret = canny[x][y - stp_y];              //#
						if( ret )
						{
							return ret;
						}
					}
					else
					{
						return NULL;
					}
				}
			}
			else
			{
				return NULL;
			}
		}
	}
	else
	{
		float a   = ady / adx;
		int stp_x = boost::math::sign( dx );
		int stp_y = boost::math::sign( dy );

		int x = p.x();
		int y = p.y();

		n = n + 1;
		e = e + a;
		x = x + stp_x;

		if( e >= 0.5f )
		{
			y = y + stp_y;
			e = e - 1.0f;
		}

		// Partie commenté dû à la différence de réponse du détecteur de contour
		/*if ( x >= 0 && x < canny.shape()[0] &&
		     y >= 0 && y < canny.shape()[1] )
		   {
		    ret = canny[x][y];
		    if ( ret )
		    {
		        return ret;
		    }
		   }
		   else
		   {
		    return NULL;
		   }*/
		n = n + 1;
		e = e + a;
		x = x + stp_x;
		if( e >= 0.5f )
		{
			y = y + stp_y;
			e = e - 1;
		}
		if( x >= 0 && x < canny.shape()[0] &&
		    y >= 0 && y < canny.shape()[1] )
		{
			ret = canny[x][y];
			if( ret )
			{
				return ret;
			}
		}
		else
		{
			return NULL;
		}

		while( n <= nmax )
		{
			n = n + 1;
			e = e + a;
			x = x + stp_x;
			if( e >= 0.5f )
			{
				y = y + stp_y;
				e = e - 1.0f;
			}

			if( x >= 0 && x < canny.shape()[0] &&
			    y >= 0 && y < canny.shape()[1] )
			{
				ret = canny[x][y];
				if( ret )
				{
					return ret;
				}
				else
				{
					if( ( x - stp_x ) >= 0 && ( x - stp_x ) < canny.shape()[0] &&
					    y >= 0 && y < canny.shape()[1] )
					{
						ret = canny[x - stp_x][y];
						if( ret )
						{
							return ret;
						}
					}
					else
					{
						return NULL;
					}
				}
			}
			else
			{
				return NULL;
			}
		}
	}
	return NULL;
}

void updateXY(const float & dx, const float & dy, int & x, int & y,  float & e, int & stpX, int & stpY)
{
	float a = std::abs(dy/dx);
	stpX = boost::math::sign<int>(dx);
	stpY = boost::math::sign<int>(dy);
	e += a;
	x += stpX;
	if (e>=0.5)
	{
		y += stpY;
		e -= 1;
	}
	return;
}

EdgePoint* gradientDirectionDescent(
        const boost::multi_array<EdgePoint*, 2> & canny, 
        const EdgePoint& p,
        int dir,
        const std::size_t nmax, 
        const cv::Mat & imgDx, 
        const cv::Mat & imgDy, 
        int thrGradient)
{
    EdgePoint* ret = NULL;
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
            ret = canny[x][y];
            if( ret )
            {
                    return ret;
            }
        }
        else
        {
                return NULL;
        }

        while( n <= nmax)
        {
            updateXY(dy,dx,y,x,e, stpY,stpX);
            CCTagFileDebug::instance().addFieldLinePoint(x, y);
            n = n+1;

            if( x >= 0 && x < canny.shape()[0] &&
                y >= 0 && y < canny.shape()[1] )
            {
                ret = canny[x][y];
                if( ret )
                {
                    return ret;
                }
                else
                {
                    if( x >= 0 && x < canny.shape()[0] &&
                        ( y - stpY ) >= 0 && ( y - stpY ) < canny.shape()[1] )
                    {
                        ret = canny[x][y - stpY];              //#
                        if( ret )
                        {
                                return ret;
                        }
                    }
                    else
                    {
                            return NULL;
                    }
                }
            }
            else
            {
                    return NULL;
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
            ret = canny[x][y];
            if( ret )
            {
                return ret;
            }
        }
        else
        {
            return NULL;
        }

        while( n <= nmax)
        {
            updateXY(dx,dy,x,y,e,stpX,stpY);
            CCTagFileDebug::instance().addFieldLinePoint(x, y);
            n = n+1;

            if( x >= 0 && x < canny.shape()[0] &&
                y >= 0 && y < canny.shape()[1] )
            {
                ret = canny[x][y];
                if( ret )
                {
                    return ret;
                }
                else
                {
                    if( ( x - stpX ) >= 0 && ( x - stpX ) < canny.shape()[0] &&
                        y >= 0 && y < canny.shape()[1] )
                    {
                        ret = canny[x - stpX][y];
                        if( ret )
                        {
                                return ret;
                        }
                    }
                    else
                    {
                        return NULL;
                    }
                }
            }
            else
            {
                return NULL;
            }
        }
    }
    return NULL;
}

} // namespace cctag
