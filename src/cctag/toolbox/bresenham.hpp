#ifndef _ROM_TOOLBOX_BRESENHAM_HPP_
#define	_ROM_TOOLBOX_BRESENHAM_HPP_

#include <boost/gil/gil_all.hpp>
#include <cctag/imageCut.hpp>

namespace rom
{
namespace toolbox
{

void bresenham( const boost::gil::gray8_view_t & sView, const rom::Point2dN<int>& p, const rom::Point2dN<float>& dir, const std::size_t nmax, ImageCut & cut );

}	
}

#endif
