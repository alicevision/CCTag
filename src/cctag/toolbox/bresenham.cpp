#include "bresenham.hpp"

#include <boost/math/special_functions/sign.hpp>

namespace cctag
{
namespace toolbox
{

void bresenham( const boost::gil::gray8_view_t & sView, const cctag::Point2dN<int>& p, const cctag::Point2dN<float>& dir, const std::size_t nmax, ImageCut & cut )
{
	float e        = 0.0f;
	float dx       = dir.x();
	float dy       = dir.y();

	float adx = std::abs( dx );
	float ady = std::abs( dy );

	std::size_t n = 0;

	cut._start = p;
	cut._stop = p;
	std::vector<double> res;
	res.reserve( 3 * nmax );

	if( ady > adx )
	{
		float a   = adx / ady;                          //#
		int stp_x = boost::math::sign( dx );
		int stp_y = boost::math::sign( dy );

		int x = p.x();
		int y = p.y();
		res.push_back( *sView.xy_at( x, y ) );

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
			res.push_back( *sView.xy_at( x, y ) );
			cut._stop = cctag::Point2dN<int>( x, y );
		}
		else
		{
			cut._imgSignal.resize( res.size() );
			std::copy( res.begin(), res.end(), cut._imgSignal.begin() );
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
				res.push_back( *sView.xy_at( x, y ) );
				cut._stop = cctag::Point2dN<int>( x, y );
			}
			else
			{
				cut._imgSignal.resize( res.size() );
				std::copy( res.begin(), res.end(), cut._imgSignal.begin() );
				return;
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
			res.push_back( *sView.xy_at( x, y ) );
			cut._stop = cctag::Point2dN<int>( x, y );
		}
		else
		{
			cut._imgSignal.resize( res.size() );
			std::copy( res.begin(), res.end(), cut._imgSignal.begin() );
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
				res.push_back( *sView.xy_at( x, y ) );
				cut._stop = cctag::Point2dN<int>( x, y );
			}
			else
			{
				cut._imgSignal.resize( res.size() );
				std::copy( res.begin(), res.end(), cut._imgSignal.begin() );
				return;
			}
		}
	}
	cut._imgSignal.resize( res.size() );
	std::copy( res.begin(), res.end(), cut._imgSignal.begin() );
}

}
}
