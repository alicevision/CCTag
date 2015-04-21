#ifndef _CCTAG_FRAME_HPP
#define	_CCTAG_FRAME_HPP

#include <string>

namespace cctag {

typedef double Time;

struct FrameTimed
{
	FrameTimed(): _frame(0), _time(0.0) {}

	FrameTimed( const std::size_t frame, const Time time )
	: _frame( frame )
	, _time( time )
	{}

	std::size_t _frame;
	Time  _time;

	inline bool operator<( const FrameTimed & f ) { return _frame < f._frame; }
	inline bool operator<=( const FrameTimed & f ) { return _frame <= f._frame; }
	inline bool operator>( const FrameTimed & f ) { return _frame > f._frame; }
	inline bool operator>=( const FrameTimed & f ) { return _frame >= f._frame; }

};

struct FrameRange
{
	FrameRange( const std::size_t r1, const std::size_t r2 ): first( r1 ), last( r2 ) {};
	FrameRange(): first(0), last(0) {};

	std::size_t first;
	std::size_t last;
};

}

#endif
