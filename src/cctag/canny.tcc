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

template<class SView,
         class CannyRGBView,
         class CannyView, 
         class GradXView,
         class GradYView>
void cannyCv(
        const SView& srcView,
        CannyRGBView& cannyRGB,
        CannyView& cannyView,
        GradXView& cannyGradX,
        GradYView& cannyGradY,
        const double thrCannyLow,
        const double thrCannyHigh)
{
  using namespace boost::gil;

  typedef pixel<typename channel_type<SView>::type, layout<gray_t> > PixelGray;
  typedef image<PixelGray, false> GrayImage;
  typedef typename GrayImage::view_t GrayView;
  typedef typename channel_type<GrayView>::type Precision;

  boost::posix_time::ptime t1a( boost::posix_time::microsec_clock::local_time() );
  cvCanny( cannyRGB, boostCv::CvImageView( srcView ).get(), thrCannyLow, thrCannyHigh );
  boost::posix_time::ptime t2a( boost::posix_time::microsec_clock::local_time() );

  // Thinning
  using namespace terry;
  using namespace terry::numeric;
  typedef Rect<std::ptrdiff_t> rect_t;
  
  rect_t srcRod( 0, 0, cannyView.width(), cannyView.height() );
  rect_t srcRodCrop1( 1, 1, cannyView.width() - 1, cannyView.height() - 1 );
  rect_t srcRodCrop2( 1, 1, cannyView.width() - 1, cannyView.height() - 1 );
  rect_t procWindowRoWCrop1 = srcRodCrop1;
  rect_t procWindowRoWCrop2 = srcRodCrop2;
  
  IPoolDataPtr dataTmp = MemoryPool::instance().allocate(
          cannyView.width() * cannyView.height() * sizeof(Precision) );
  
  GrayView view_tmp = interleaved_view(
          srcView.width(), srcView.height(),
          (PixelGray*)dataTmp->data(), cannyView.width() * sizeof(Precision) );
  
  PixelGray pixelZero;
  pixel_zeros_t<PixelGray>()( pixelZero );
  fill_pixels( view_tmp, pixelZero );
  
  algorithm::transform_pixels_locator(
        cannyView, srcRod,
        view_tmp, srcRod,
        procWindowRoWCrop1,
        terry::filter::thinning::pixel_locator_thinning_t<CannyView, GrayView>( cannyView, terry::filter::thinning::lutthin1 ) );

  fill_pixels( cannyView, pixelZero );
  algorithm::transform_pixels_locator(
        view_tmp, srcRod,
        cannyView, srcRod,
        procWindowRoWCrop2,
        terry::filter::thinning::pixel_locator_thinning_t<GrayView, CannyView>( view_tmp, terry::filter::thinning::lutthin2 ) );
}

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

