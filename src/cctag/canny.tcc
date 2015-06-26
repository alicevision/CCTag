#include <cctag/progBase/MemoryPool.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/boostCv/cvImage.hpp>

#include <terry/filter/thinning.hpp>
#include <terry/sampler/sampler.hpp>
#include <terry/numeric/init.hpp>

#include <opencv/cv.h>

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/typedefs.hpp>

#include <boost/gil/image_view_factory.hpp>

namespace cctag
{

template<class CView, class DXView, class DYView>
void edgesPointsFromCanny(
        std::vector<EdgePoint>& points,
        EdgePointsImage & edgePointsMap,
        CView & cannyView,
        DXView & dx,
        DYView & dy )
{
  using namespace boost::gil;

  edgePointsMap.resize( boost::extents[cannyView.width()][cannyView.height()] );
  std::fill( edgePointsMap.origin(), edgePointsMap.origin() + edgePointsMap.size(), (EdgePoint*)NULL );

  // todo@Lilian: is this upper bound correct ?
  points.reserve( cannyView.width() * cannyView.height() / 2 );

  for( int y = 0 ; y < cannyView.height(); ++y )
  {
    typename CView::x_iterator itc = cannyView.row_begin( y );
    typename DXView::x_iterator itDx = dx.row_begin( y );
    typename DYView::x_iterator itDy = dy.row_begin( y );
    for( int x = 0 ; x < cannyView.width(); ++x )
    {
      if ( (*itc)[0] )
      {
        points.push_back( EdgePoint( x, y, (*itDx)[0], (*itDy)[0] ) );
        EdgePoint* p = &points.back();
        edgePointsMap[x][y] = p;
      }
      ++itc;
      ++itDx;
      ++itDy;
    }
  }

}

} // namespace cctag

