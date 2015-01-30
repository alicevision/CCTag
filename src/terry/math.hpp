#ifndef _TERRY_MATH_HPP_
#define	_TERRY_MATH_HPP_

#include "basic_colors.hpp"

namespace terry {

/**
 * @brief Compute min & max value from a view
 *
 * @param[in]   view     Source view
 * @param[out]  max      maximum image value
 * @param[out]  min      minimum image value
 *
 */
template <typename View, typename T>
void maxmin( const View& view, T& max, T& min )
{
	using namespace boost::gil;
	typedef typename View::x_iterator iterator;
	typedef typename channel_type<View>::type dPix_t;
	const int nc = view.num_channels();
	int w        = view.width();
	int h        = view.height();
	max = min = view( 0, 0 )[0];
	for( int y = 0; y < h; ++y )
	{
		iterator view_it = view.row_begin( y );
		for( int x = 0; x < w; ++x )
		{
			for( int c = 0; c < nc; c++ )
			{
				const dPix_t val = ( *view_it )[c];
				if( val > max )
				{
					max = val;
				}
				else if( val < min )
				{
					min = val;
				}
			}
			++view_it;
		}
	}
}

/**
 * @brief Normalize a view (using contrast streching)
 *
 * @param[in, out]  dst     Source and destination view
 * @param[in]       a       lower limit
 * @param[in]       b       upper limit
 * @return Return the normalized image
 */
template <class S_VIEW, class D_VIEW, typename T>
D_VIEW& normalize( const S_VIEW& src, D_VIEW& dst, const T a, const T b )
{
	using namespace boost::gil;
	typedef typename S_VIEW::x_iterator sIterator;
	typedef typename D_VIEW::x_iterator dIterator;
	typedef typename channel_type<D_VIEW>::type dPix_t;
	dPix_t m, M;
	maxmin( dst, M, m );
	const float fm = m, fM = M;
	int w          = dst.width();
	int h          = dst.height();

	if( m == M )
	{
		fill_black( dst );
	}
	else if( m != a || M != b )
	{
		int nc = dst.num_channels();
		for( int y = 0; y < h; ++y )
		{
			sIterator src_it = src.row_begin( y );
			dIterator dst_it = dst.row_begin( y );
			for( int x = 0; x < w; ++x )
			{
				for( int c = 0; c < nc; c++ )
				{
					( *dst_it )[c] = (dPix_t)( ( ( *src_it )[c] - fm ) / ( fM - fm ) * ( b - a ) + a );
				}
				++dst_it;
				++src_it;
			}
		}
	}
	return dst;
}

template <class S_VIEW, class D_VIEW, typename T>
D_VIEW& multiply( const S_VIEW& src, D_VIEW& dst, const T factor )
{
	using namespace boost::gil;
	typedef typename S_VIEW::x_iterator sIterator;
	typedef typename D_VIEW::x_iterator dIterator;
	typedef typename channel_type<D_VIEW>::type dPix_t;

	const int nc = src.num_channels();
	const int w  = src.width();
	const int h  = src.height();
	int x, y, c;
	for( y = 0; y < h; y++ )
	{
		sIterator src_it = src.row_begin( y );
		dIterator dst_it = dst.row_begin( y );
		for( x = 0; x < w; x++ )
		{
			for( c = 0; c < nc; c++ )
			{
				( *dst_it )[c] = (dPix_t)( ( *src_it )[c] * factor );
			}
			++src_it;
			++dst_it;
		}
	}
	return dst;
}

}

#endif

