#ifndef _TERRY_FILTER_FLOODFILL_HPP_
#define _TERRY_FILTER_FLOODFILL_HPP_

#include <terry/globals.hpp>
#include <terry/draw/fill.hpp>
#include <terry/basic_colors.hpp>
#include <terry/channel.hpp>
#include <terry/math/Rect.hpp>
#include <terry/numeric/minmax.hpp>
#include <terry/algorithm/transform_pixels.hpp>

#include <queue>
#include <list>


namespace terry {
namespace filter {
namespace floodFill {

/**
 * @brief fill all pixels respecting the @p condition in a range of pixels (on the same line or with an 1d_traversable image).
 */
template<class SIterator, class DIterator, class DPixel, class Test>
GIL_FORCEINLINE
void fill_pixels_range_if( SIterator srcBegin, const SIterator& srcEnd,
                    DIterator dstBegin, const DPixel& value,
                    const Test& condition )
{
	do
	{
		if( condition( (*srcBegin)[0] ) )
		{
#ifdef TERRY_DEBUG_FLOODFILL
			if( (*dstBegin)[0] != value[0] )
			{
				*dstBegin = value;
			}
#else
			*dstBegin = value;
#endif
		}
		++srcBegin;
		++dstBegin;
	}
	while( srcBegin != srcEnd );
}

struct Connexity4
{
	static const std::ssize_t x = 0;
};

struct Connexity8
{
	static const std::ssize_t x = 1;
};

template<typename T>
struct IsUpper
{
	const T _threshold;
	IsUpper( const T& threshold )
	: _threshold(threshold)
	{}
	GIL_FORCEINLINE
	bool operator()( const T& p ) const
	{
		return (p >= _threshold);
	}
};


typedef enum
{
	eDirectionAbove = 0,
	eDirectionBellow = 1
} EDirection;

GIL_FORCEINLINE
EDirection invertDirection( const EDirection d )
{
	return d == eDirectionAbove ? eDirectionBellow : eDirectionAbove;
}

/**
 * @brief A range of pixels in a line to propagate.
 */
template<class SView, class DView>
struct FloodElement
{
	typedef typename SView::xy_locator SLocator;
	typedef typename DView::xy_locator DLocator;
	typedef typename SView::value_type SPixel;
	typedef typename DView::value_type DPixel;

	SLocator _srcBegin; //< locator on the first pixel of the source image of the range to propagate
	SLocator _srcEnd;   //< like end() in the stl, this is the pixel after the last pixel of the range
	DLocator _dstBegin; //< locator on the first pixel of the dest image
	std::ssize_t _xBegin; //< x coordinate of the first pixel
	std::ssize_t _xEnd; //< x coordinate of the pixel after the last pixel of the range
	std::ssize_t _y; //< y coordinate
	EDirection _direction; //< the direction of the propagation to apply
};

/**
 * @brief Flood fill an image, with 2 conditions.
 * Fill all pixels respecting the soft condition if connected with a pixel respecting the strong condition.
 *
 * @param[in] srcView input image
 * @param[in] srcRod input image ROD
 * @param[out] dstView input image
 * @param[in] dstRod output image ROD
 * @param[in] procWindow region to process
 * @param[in] strongCondition functor with the strong test
 * @param[in] softCondition functor with the soft test
 *
 * @remark Implementation is based on standard filling algorithms. So we use ranges by line (x axis),
 * and check connections between these x ranges and possible x ranges in the above or bellow lines.
 */
template<class Connexity, class StrongTest, class SoftTest, class SView, class DView, template<class> class Allocator>
void flood_fill( const SView& srcView, const Rect<std::ssize_t>& srcRod,
                 DView& dstView, const Rect<std::ssize_t>& dstRod,
                                 const Rect<std::ssize_t>& procWindow,
				 const StrongTest& strongTest, const SoftTest& softTest )
{
	using namespace terry;
	using namespace terry::draw;
	typedef typename SView::xy_locator SLocator;
	typedef typename DView::xy_locator DLocator;
	typedef typename SView::value_type SPixel;
	typedef typename DView::value_type DPixel;
	typedef typename boost::gil::channel_type<SPixel>::type SChannel;
	typedef typename terry::channel_base_type<SChannel>::type SType;
	typedef typename SLocator::cached_location_t SCachedLocation;
	typedef typename DLocator::cached_location_t DCachedLocation;

	typedef FloodElement<SView, DView> FloodElem;

	static const DPixel white = get_white<DPixel>();

#ifdef TERRY_DEBUG_FLOODFILL
	DPixel red = get_black<DPixel>();
	red[0] = 1.0;
	DPixel yellow = get_black<DPixel>();
	yellow[0] = 1.0;
	yellow[1] = 1.0;
#endif
	
	const Rect<std::ssize_t> rod = rectanglesIntersection( srcRod, dstRod );

	const std::size_t procWidth = (procWindow.x2 - procWindow.x1);
	const std::size_t halfProcWidth = procWidth / 2;

	const SLocator sloc_ref( srcView.xy_at(0,0) );
	const SLocator dloc_ref( dstView.xy_at(0,0) );

	std::vector<FloodElem, Allocator<FloodElem> > propagation;
	propagation.reserve( halfProcWidth );

	SLocator src_loc = srcView.xy_at( procWindow.x1 - srcRod.x1, procWindow.y1 - srcRod.y1 );
	DLocator dst_loc = dstView.xy_at( procWindow.x1 - dstRod.x1, procWindow.y1 - dstRod.y1 );

//	// LT CT RT
//	// LC    RC
//	// LB CB RB
//	const SCachedLocation sLT( src_loc.cache_location(-Connectivity::x,-1) );
//	const SCachedLocation sRT( src_loc.cache_location( Connectivity::x,-1) );
//	const SCachedLocation sLB( src_loc.cache_location(-Connectivity::x, 1) );
//	const SCachedLocation sRB( src_loc.cache_location( Connectivity::x, 1) );
//
//	const DCachedLocation dLT( dst_loc.cache_location(-Connectivity::x,-1) );
//	const DCachedLocation dRT( dst_loc.cache_location( Connectivity::x,-1) );
//	const DCachedLocation dLB( dst_loc.cache_location(-Connectivity::x, 1) );
//	const DCachedLocation dRB( dst_loc.cache_location( Connectivity::x, 1) );

	boost::gil::point2<std::ptrdiff_t> endToBegin( -(procWindow.x2-procWindow.x1),1);
	boost::gil::point2<std::ptrdiff_t> nextLine(0,1);
	boost::gil::point2<std::ptrdiff_t> previousLine( 0,-1);
		
	for( std::ssize_t y = procWindow.y1;
	     y < procWindow.y2;
	     ++y )
	{
		{
			std::size_t xLastBegin = 0;
			SLocator srcLastBegin;
			DLocator dstLastBegin;
			bool inside = false;
			bool containsStrong = false;

			for( std::ssize_t x = procWindow.x1;
				 x < procWindow.x2;
				 ++x, ++dst_loc.x(), ++src_loc.x() )
			{
				if( (*dst_loc)[0] == white[0] )
				{
					if( ! inside )
					{
						xLastBegin = x;
						srcLastBegin = src_loc;
						dstLastBegin = dst_loc;
						inside = true;
					}
					containsStrong = true;
				}
				else if( softTest( (*src_loc)[0] ) )
				{
					if( ! inside )
					{
						xLastBegin = x;
						srcLastBegin = src_loc;
						dstLastBegin = dst_loc;
						inside = true;
					}
					if( ! containsStrong )
					{
						if( strongTest( (*src_loc)[0] ) )
						{
							containsStrong = true;
						}
					}
				}
				else
				{
					// do the same thing at the end of a line
					if( /*inside &&*/ containsStrong )
					{
						// visit line above
						FloodElem el;
						el._srcBegin = srcLastBegin;
						el._srcEnd = src_loc;
						el._dstBegin = dstLastBegin;
						el._xBegin = xLastBegin;
						el._xEnd = x;
						el._y = y;
						el._direction = eDirectionAbove;
						propagation.push_back( el );

						// current line
						// fill output from first to last
						fill_pixels_range( dstLastBegin.x(), dst_loc.x(), white );

						// visit line bellow
						if( y < rod.y2  )
						{
	//						fill_range_if( srcLastBegin[sLB], src_loc[sRB], dstLastBegin[dLB], white, softTest );
							// fill line bellow for current range if respect softTest
							fill_pixels_range_if( srcLastBegin.x_at(-Connexity::x,1), src_loc.x_at(Connexity::x,1),
										   dstLastBegin.x_at(-Connexity::x,1),
										   white, softTest );
						}
					}
					// re-init values
					inside = false;
					containsStrong = false;
				}
			}
			// if all the last pixels are a group of detected pixels
			if( /*inside &&*/ containsStrong )
			{
				// visit line above
				FloodElem el;
				el._srcBegin = srcLastBegin;
				el._srcEnd = src_loc;
				el._dstBegin = dstLastBegin;
				el._xBegin = xLastBegin;
				el._xEnd = procWindow.x2;
				el._y = y;
				el._direction = eDirectionAbove;
				propagation.push_back( el );

				// current line
				// fill output from first to last
				fill_pixels_range( dstLastBegin.x(), dst_loc.x(), white );

				// visit line bellow
				if( y < rod.y2  )
				{
					// fill line bellow for current range if respect softTest
					fill_pixels_range_if( srcLastBegin.x_at(-Connexity::x,1), src_loc.x_at(Connexity::x,1),
								   dstLastBegin.x_at(-Connexity::x,1),
								   white, softTest );
				}
			}
		}
		// move to the beginning of the next line
		dst_loc += endToBegin;
		src_loc += endToBegin;

		// do the propagation on the above lines
		while( !propagation.empty() )
		{
			FloodElem iElem = propagation.back();
			propagation.pop_back();

			// after having been up, we can get down up to next line (at the maximum)
			if( iElem._direction == eDirectionBellow &&
			    iElem._y > y+1 )
				continue;
			if( (iElem._direction == eDirectionAbove && iElem._y < procWindow.y1 ) ||
			    (iElem._direction == eDirectionBellow && iElem._y >= procWindow.y2 ) )
				continue;

			// i: the "input" range
			// g: is the full range, where we can found f elements
			// f: ranges "founded" (connected with i range)
			// s: futur range to "search"
			//
			// first step:
			// [ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg] // current line to apply the propagation
			// [sssssssssssss].[iiiiiiiiiiiiiiiiiiiiiiiiiiii].[ssssssssssssss]
			//                              ^
			//                              |  direction: above
			// 
			// example 1:
			// [sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss] // search in the same direction
			// [fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff] // current line to apply the propagation
			// [sssssssssssss].[iiiiiiiiiiiiiiiiiiiiiiiiiiii].[ssssssssssssss] // search in the opposite direction
			//                              ^
			//                              |  direction: above
			//
			// example 2:
			// [sssssssssssssss].....[sssss]...[sss]..[ssssssssssssssssssssss] // search in the same direction
			// [fffffffffffffff].....[fffff]...[fff]..[ffffffffffffffffffffff] // current line to apply the propagation
			// [sssssssssssss].[iiiiiiiiiiiiiiiiiiiiiiiiiiii].[ssssssssssssss] // search in the opposite direction
			//                              ^
			//                              |  direction: above
			//
			// example 3:
			// .....................[sssss]....[sss]..[ssssssssssssssssssssss] // search in the same direction
			// .....................[fffff]....[fff]..[ffffffffffffffffffffff] // current line to apply the propagation
			// ................[iiiiiiiiiiiiiiiiiiiiiiiiiiii].[ssssssssssssss] // search in the opposite direction
			//                              ^
			//                              |  direction: above

			FloodElem gElem = iElem;

			if( gElem._direction == eDirectionAbove )
			{
				gElem._srcBegin += previousLine;
				gElem._srcEnd   += previousLine;
				gElem._dstBegin += previousLine;
				--(gElem._y);
			}
			else
			{
				gElem._srcBegin += nextLine;
				gElem._srcEnd   += nextLine;
				gElem._dstBegin += nextLine;
				++(gElem._y);
			}

			// x propagation
			// left
			if( Connexity::x ||
			    softTest( (*gElem._srcBegin)[0] ) ) // check first
			{
				while( gElem._xBegin-1 >= rod.x1 &&
					   softTest( (*(gElem._srcBegin.x()-1))[0] ) )
				{
					--(gElem._xBegin);
					--(gElem._srcBegin.x());
					--(gElem._dstBegin.x());
				}
			}
			// right
			if( Connexity::x ||
			    softTest( (*(gElem._srcEnd.x()-1))[0] ) ) // check last
			{
				while( gElem._xEnd < rod.x2 &&
					   softTest( (*gElem._srcEnd)[0] ) )
				{
					++(gElem._xEnd);
					++(gElem._srcEnd.x());
				}
			}

			// new sub-ranges
			{
				bool inside = false;
				bool modified = false;
				std::list<FloodElem> localPropagation;
				FloodElem localElem = gElem;
				localElem._srcEnd = localElem._srcBegin;
				DLocator dstEnd = localElem._dstBegin;

				for( std::ssize_t xx = gElem._xBegin;
					 xx < gElem._xEnd;
					 ++xx )
				{
					if( softTest( (*localElem._srcEnd)[0] ) )
					{
						if( ! inside )
						{
							localElem._srcBegin = localElem._srcEnd;
							localElem._dstBegin = dstEnd;
							localElem._xBegin = xx;
							inside = true;
						}
						if( !modified )
						{
							if( (*dstEnd)[0] != white[0] )
							{
								modified = true;
							}
						}
#ifdef TERRY_DEBUG_FLOODFILL
						// use one color for each case to debug
						if( (*dstEnd)[0] != white[0] )
						{
							if( iElem._direction == eDirectionAbove )
								*dstEnd = red;
							else
								*dstEnd = yellow;
						}
#else
						*dstEnd = white;
#endif
					}
					else
					{
						if( inside )
						{
							localElem._xEnd = xx;
							// if we just have modified a pixel in the output,
							// possibly we have to propagate something
							if( modified )
							{
								// propagation in the same direction
								localPropagation.push_back( localElem );
							}
							inside = false;
						}
						modified = false;
					}
					++localElem._srcEnd.x();
					++dstEnd.x();
				}
				if( inside )
				{
					localElem._xEnd = gElem._xEnd;
					if( modified )
					{
						// propagation in the same direction
						localPropagation.push_back( localElem );
					}
				}

				if( ! localPropagation.empty() )
				{
					// propagation in the opposite direction
					if( localPropagation.front()._xBegin - 1 + Connexity::x < iElem._xBegin )
					{
						FloodElem leftElem = iElem; // same line as source
						leftElem._direction = invertDirection( iElem._direction );
						leftElem._xBegin = localPropagation.front()._xBegin;
						leftElem._xEnd = iElem._xBegin - 1;
						leftElem._srcBegin = localPropagation.front()._srcBegin;
						leftElem._dstBegin = localPropagation.front()._dstBegin;
						leftElem._srcEnd = iElem._srcEnd.xy_at( -1, 0 );
						propagation.push_back( leftElem );
					}
					if( localPropagation.back()._xEnd + 1 - Connexity::x > iElem._xEnd )
					{
						FloodElem rightElem = iElem; // same line as source
						rightElem._direction = invertDirection( iElem._direction );
						rightElem._xBegin = iElem._xEnd + 1;
						std::size_t shift = rightElem._xBegin - localPropagation.back()._xBegin;
						rightElem._srcBegin = localPropagation.back()._srcBegin.xy_at( shift, 0 );
						rightElem._dstBegin = localPropagation.back()._dstBegin.xy_at( shift, 0 );
						rightElem._xEnd = localPropagation.back()._xEnd;
						rightElem._srcEnd = rightElem._srcBegin.xy_at( rightElem._xEnd - rightElem._xBegin, 0 );
						propagation.push_back( rightElem );
					}

					propagation.insert( propagation.end(), localPropagation.begin(), localPropagation.end() );
				}
			}
		}
	}
}


/**
 * @brief Simplest implementation of flood fill algorithm.
 * @see flood_fill, faster implementation of the same algorithm
 * @remark not in production (only use for debugging)
 */
template<class StrongTest, class SoftTest, class SView, class DView>
void flood_fill_bruteForce( const SView& srcView, const Rect<std::ssize_t>& srcRod,
                                 DView& dstView, const Rect<std::ssize_t>& dstRod,
                                 const Rect<std::ssize_t>& procWindow,
				 const StrongTest& strongTest, const SoftTest& softTest )
{
    typedef typename SView::value_type SPixel;
    typedef typename boost::gil::channel_type<SView>::type SChannel;
	typedef typename SView::iterator SIterator;
    typedef typename DView::value_type DPixel;
    typedef typename boost::gil::channel_type<DView>::type DChannel;
	typedef typename DView::iterator DIterator;

	typedef boost::gil::point2<std::ptrdiff_t> Point2;

	static const std::size_t gradMax = 0;
	static const std::size_t lower = 0;
	static const std::size_t upper = 1;
	static const std::size_t flooding = 2;

	Rect<int> procWindowOutput = translateRegion( procWindow, dstRod );
	boost::gil::point2<int> procWindowSize( procWindow.x2 - procWindow.x1, procWindow.y2 - procWindow.y1 );
	Rect<int> rectLimit = rectangleReduce(procWindow, 1);

	const Point2 nextLine( -procWindowSize.x, 1 );
	typename DView::xy_locator dst_loc = dstView.xy_at( procWindowOutput.x1, procWindowOutput.y1 );
	typename DView::xy_locator src_loc = srcView.xy_at( procWindow.x1 - srcRod.x1, procWindow.y1 - srcRod.y1 );

	for( int y = procWindow.y1;
			 y < procWindow.y2;
			 ++y )
	{
		for( int x = procWindow.x1;
			 x < procWindow.x2;
			 ++x, ++dst_loc.x(), ++src_loc.x() )
		{
			if( softTest( (*src_loc)[gradMax] ) )
			{
				(*dst_loc)[lower] = boost::gil::channel_traits<DChannel>::max_value();
				if( strongTest( (*src_loc)[gradMax] ) )
				{
					(*dst_loc)[upper] = boost::gil::channel_traits<DChannel>::max_value();
					std::queue<Point2> fifo; ///< @todo tuttle: use host allocator
					fifo.push( Point2( x, y ) );
					while( !fifo.empty() )
					{
						const Point2 p = fifo.front();
						fifo.pop();
						for( int dy = -1; dy < 2; ++dy )
						{
							for( int dx = -1; dx < 2; ++dx )
							{
								DIterator pix = dstView.at( p.x+dx - dstRod.x1, p.y+dy - dstRod.y1 );
								SIterator spix = srcView.at( p.x+dx - srcRod.x1, p.y+dy - srcRod.y1 );
								if( softTest( (*spix)[gradMax] ) &&
									(*pix)[flooding] != boost::gil::channel_traits<DChannel>::max_value() )
								{
									(*pix)[flooding] = boost::gil::channel_traits<DChannel>::max_value();
									if( ! strongTest( (*spix)[gradMax] ) )
									{
										Point2 np( p.x+dx, p.y+dy );
										// inside a subwindow of the rendering,
										// we can't append border pixels
										if( pointInRect( np, rectLimit ) )
											fifo.push( np );
									}
								}
							}
						}
					}
				}
			}
//			else
//			{
//				(*dst_loc)[upper] = boost::gil::channel_traits<DChannel>::max_value();
//			}
		}
		dst_loc += nextLine;
		src_loc += nextLine;
	}

//	if( _params._fillAllChannels )
//	{
	DView tmp_dst = subimage_view( dstView, procWindowOutput.x1, procWindowOutput.y1,
											  procWindowSize.x, procWindowSize.y );
	boost::gil::copy_pixels( boost::gil::kth_channel_view<flooding>(tmp_dst), boost::gil::kth_channel_view<lower>(tmp_dst) );
	boost::gil::copy_pixels( boost::gil::kth_channel_view<flooding>(tmp_dst), boost::gil::kth_channel_view<upper>(tmp_dst) );
	boost::gil::copy_pixels( boost::gil::kth_channel_view<flooding>(tmp_dst), boost::gil::kth_channel_view<3>(tmp_dst) );
//	}
}

}


template<template<class> class Allocator, class SView, class DView>
void applyFloodFill(
	const SView& srcView,
	      DView& dstView,
	const double lowerThres, const double upperThres )
{
	using namespace boost::gil;
	using namespace terry::numeric;

	typedef double Scalar;
	typedef typename SView::value_type SPixel;
	typedef typename DView::value_type DPixel;

	typedef kth_channel_view_type<0,SView> LocalView;
	typename LocalView::type localView( LocalView::make(srcView) );
	pixel_minmax_by_channel_t<typename LocalView::type::value_type> minmax( localView(0,0) );

	terry::algorithm::transform_pixels(
		localView,
		minmax );
	
	const bool isConstantImage = ( minmax.max[0] == minmax.min[0] );
	const double lowerThresR = (lowerThres * (minmax.max[0]-minmax.min[0])) + minmax.min[0];
	const double upperThresR = (upperThres * (minmax.max[0]-minmax.min[0])) + minmax.min[0];

	terry::draw::fill_pixels( dstView, get_black<DPixel>() );
	
	if( isConstantImage )
		return;
	
	floodFill::flood_fill<floodFill::Connexity4, floodFill::IsUpper<Scalar>, floodFill::IsUpper<Scalar>, SView, DView, Allocator>(
				srcView, getBounds<std::ptrdiff_t>(srcView),
				dstView, getBounds<std::ptrdiff_t>(dstView),
				rectangleReduce( getBounds<std::ptrdiff_t>(dstView), 1 ),
				floodFill::IsUpper<Scalar>(upperThresR),
				floodFill::IsUpper<Scalar>(lowerThresR)
				);
}


}
}

#endif
