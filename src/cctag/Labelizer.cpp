#include "Labelizer.hpp"
#include "Label.hpp"

#include <boost/gil/locator.hpp>


namespace cctag {
namespace vision {

Labelizer::Labelizer()
{}

Labelizer::~Labelizer()
{}

void Labelizer::followContour( Label* label, const int x, const int y, const boost::gil::gray16s_view_t& dxVw, const boost::gil::gray16s_view_t& dyVw, LabelizedImage& labelsMap )
{
	if( x >= 0 && x < labelsMap.shape()[0] && y >= 0 && y < labelsMap.shape()[1] && !labelsMap[x][y] )
	{
		// Add current point
		label->push_back( LabelEdgePoint( x, y, *dxVw.xy_at( x, y ), *dyVw.xy_at( x, y ), label ) );
		labelsMap[x][y] = &label->back();

		// Process next point
		static int addX[] = { 1,  1,  0, -1, -1, -1,  0,  1 };
		static int addY[] = { 0,  1,  1,  1,  0, -1, -1, -1 };
		std::size_t k     = 0;

		for( k = 0; k < 8; ++k )
		{
			followContour( label, x + addX[k], y + addY[k], dxVw, dyVw, labelsMap );
		}
	}
}

}
}
