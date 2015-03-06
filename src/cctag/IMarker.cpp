#include "IMarker.hpp"

namespace cctag {
namespace vision {
namespace marker {

IMarker::IMarker()
	: _quality( 0 )
{}

IMarker::IMarker( const IMarker & m )
	: _centerImg( m._centerImg )
	, _centerScene( m._centerScene )
	, _quality( m._quality )
{}

IMarker::~IMarker()
{}


}
}
}
