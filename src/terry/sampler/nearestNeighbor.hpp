#ifndef _TERRY_SAMPLER_NEARESTNEIGHBOR_HPP_
#define _TERRY_SAMPLER_NEARESTNEIGHBOR_HPP_

#include "details.hpp"
#include <terry/basic_colors.hpp>
#include <iostream>

namespace terry {
using namespace boost::gil;
namespace sampler {

struct nearest_neighbor_sampler {};

template <typename DstP, typename SrcView, typename F>
bool sample( nearest_neighbor_sampler, const SrcView& src, const point2<F>& p, DstP& result, const EParamFilterOutOfImage outOfImageProcess )
{
	typedef typename SrcView::value_type SrcP;
	point2<std::ptrdiff_t> center( iround( p ) );

	// if we are outside the image
	if( center.x < 0 )
	{
		switch( outOfImageProcess )
		{
			case eParamFilterOutBlack :
			{
				result = get_black<DstP>();
				return true;
			}
			case eParamFilterOutTransparency :
			{
				result = SrcP(0);
				return true;
			}
			case eParamFilterOutCopy :
			{
				center.x = 0.0;
				break;
			}
			case eParamFilterOutMirror :
			{
				int value = ( - 1.0 * center.x / src.width() );
				int idx = center.x + value * src.width();

				if( value % 2 == 0 ) // even - mirrored image
				{
					center.x = - idx - 1.0;
					break;
				}
				else // odd - displaced image
				{
					center.x = src.width() - 1.0 + idx ;
					break;
				}
			}
		}
	}
	if( center.x > src.width() - 1.0 )
	{
		switch( outOfImageProcess )
		{
			case eParamFilterOutBlack :
			{
				result = get_black<DstP>();
				return true;
			}
			case eParamFilterOutTransparency :
			{
				result = SrcP(0);
				return true;
			}
			case eParamFilterOutCopy :
			{
				center.x = src.width() - 1.0;
				break;
			}
			case eParamFilterOutMirror :
			{
				int value =  center.x / src.width();
				int idx = center.x - ( value + 1.0 ) * src.width();

				if( value % 2 == 0 ) // even - mirrored image
				{
					center.x = src.width() - 1.0 + idx ;
					break;
				}
				else // odd - displaced image
				{
					center.x = - idx - 1.0;
					break;
				}
			}
		}
	}
	if( center.y < 0 )
	{
		switch( outOfImageProcess )
		{
			case eParamFilterOutBlack :
			{
				result = get_black<DstP>();
				return true;
			}
			case eParamFilterOutTransparency :
			{
				result = SrcP(0);
				return true;
			}
			case eParamFilterOutCopy :
			{
				center.y = 0;
				break;
			}
			case eParamFilterOutMirror :
			{
				int value = ( - 1.0 * center.y / src.height() );
				int idx = center.y + value * src.height();

				if( value % 2 == 0 ) // even - mirrored image
				{
					center.y = - idx - 1.0;
					break;
				}
				else // odd - displaced image
				{
					center.y = src.height() - 1.0 + idx ;
					break;
				}
			}
		}
	}

	if( center.y > src.height() - 1.0 )
	{
		switch( outOfImageProcess )
		{
			case eParamFilterOutBlack :
			{
				result = get_black<DstP>();
				return true;
			}
			case eParamFilterOutTransparency :
			{
				result = SrcP(0);
				return true;
			}
			case eParamFilterOutCopy :
			{
				center.y = src.height() - 1.0;
				break;
			}
			case eParamFilterOutMirror :
			{
				int value =  center.y / src.height();
				int idx = center.y - ( value + 1.0 ) * src.height();

				if( value % 2 == 0 ) // even - mirrored image
				{
					center.y = src.height() - 1.0 + idx ;
					break;
				}
				else // odd - displaced image
				{
					center.y = - idx - 1.0;
					break;
				}
			}
		}
	}

	result = src( center.x, center.y );
	return true;
}

}
}

#endif

