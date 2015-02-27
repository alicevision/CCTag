#ifndef _POPART_VISION_CCTAG_CANNY_HPP_
#define _POPART_VISION_CCTAG_CANNY_HPP_

#include "types.hpp"

namespace popart
{
namespace vision
{
class EdgePoint;
}
}

#ifdef ROM_USE_TUTTLE
#include <tuttle/host/Graph.hpp>
#include <tuttle/host/InputBufferNode.hpp>
#else
#include <opencv2/core/types_c.h>
#endif

#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <vector>

namespace popart
{
namespace vision
{
namespace marker
{
namespace cctag
{


#ifdef ROM_USE_TUTTLE

void createCannyGraph( tuttle::host::Graph & _canny, tuttle::host::InputBufferNode* & _cannyInputBuffer, tuttle::host::Graph::Node* & cannyOutput, tuttle::host::Graph::Node* & sobelOutput );

template<class SView, class CannyView, class GradXView, class GradYView>
void cannyTuttle( std::vector<memory::CACHE_ELEMENT>& datas, const SView& svw, CannyView& cannyView, GradXView& cannyGradX, GradYView& cannyGradY, const double thrCannyLow, const double thrCannyHigh );

#else

template<class SView, class CannyRGBView, class CannyView, class GradXView, class GradYView>
void cannyCv( const SView& svw, CannyRGBView& cannyRGB, CannyView& cannyView, GradXView& cannyGradX, GradYView& cannyGradY, const double thrCannyLow, const double thrCannyHigh );

void cvCanny( boost::gil::rgb32f_view_t & cannyView, const IplImage* srcImage, const double thrLow, const double thrHigh );

#endif

template<class CView, class DXView, class DYView>
void edgesPointsFromCanny( std::vector<EdgePoint>& points, EdgePointsImage & edgesMap, CView & cvw, DXView & dx, DYView & dy );


}
}
}
}

#include "canny.tcc"

#endif

