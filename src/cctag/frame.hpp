#ifndef _CCTAG_FRAME_HPP
#define	_CCTAG_FRAME_HPP

#include <string>

namespace cctag {

typedef std::size_t FrameId;
typedef double Time;

struct FrameTimed
{
	FrameTimed(): _frame(0), _time(0.0) {}

	FrameTimed( const FrameId frame, const Time time )
	: _frame( frame )
	, _time( time )
	{}

	FrameId _frame;
	Time  _time;

	inline bool operator<( const FrameTimed & f ) { return _frame < f._frame; }
	inline bool operator<=( const FrameTimed & f ) { return _frame <= f._frame; }
	inline bool operator>( const FrameTimed & f ) { return _frame > f._frame; }
	inline bool operator>=( const FrameTimed & f ) { return _frame >= f._frame; }

};

struct FrameRange
{
	FrameRange( const FrameId r1, const FrameId r2 ): first( r1 ), last( r2 ) {};
	FrameRange(): first(0), last(0) {};

	FrameId first;
	FrameId last;
};

}

#endif
