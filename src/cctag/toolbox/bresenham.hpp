#ifndef _CCTAG_TOOLBOX_BRESENHAM_HPP_
#define	_CCTAG_TOOLBOX_BRESENHAM_HPP_

#include <boost/gil/gil_all.hpp>
#include <cctag/ImageCut.hpp>

namespace cctag
{
namespace toolbox
{

void bresenham( const boost::gil::gray8_view_t & sView, const cctag::Point2dN<int>& p, const cctag::Point2dN<float>& dir, const std::size_t nmax, ImageCut & cut );

}	
}

#endif
