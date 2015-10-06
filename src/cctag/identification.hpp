#ifndef _CCTAG_CCTAG_IDENTIFICATION_HPP_
#define	_CCTAG_CCTAG_IDENTIFICATION_HPP_

#include <cctag/SubPixEdgeOptimizer.hpp>
#include <cctag/ImageCenterOptimizer.hpp>
#include <cctag/LMImageCenterOptimizer.hpp>
#include <cctag/visualDebug.hpp>
#include <cctag/ellipseGrowing.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/distance.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/algebra/eig.hpp>
#include <cctag/algebra/svd.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/boostCv/cvImage.hpp>

#include <terry/sampler/all.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/foreach.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <vector>

namespace cctag {

enum NeighborType {
  GRID,
  CIRCULAR,
  COV
};
  
template<typename VecT>
typename VecT::value_type computeMedian( const VecT& vec )
{
  typedef typename VecT::value_type T;
  //BOOST_ASSERT( vec.size() > 0 );

  const std::size_t s = (vec.size() / 2) + 1;
  //std::cout << "s: " << s << std::endl;

  std::vector<T> output( s );
  // sort the first half of the data +1 element
  std::partial_sort_copy( vec.begin(), vec.end(), output.begin(), output.end() );

  const bool isOdd = vec.size() % 2; // impair todo@Lilian: change to bit operation

  if( isOdd )
  {
    // if the size of input datas is odd,
    // the median is the element at the center
    // output contains half of the datas, so it's the last element.
    return output.back();
  }

  // the size of the data is even, so the median is the mean of the
  // 2 elements at the center.

  return ( output[s-1] + output[s-2] ) / 2.0;
}

typedef std::vector< std::vector<double> > RadiusRatioBank;
typedef std::vector< std::pair< cctag::Point2dN<double>, cctag::ImageCut > > CutSelectionVec;

inline double dis( const double sig, const double val, const double mub, const double muw, const double varSubS )
{
  if ( val == -1 )
  {
          return boost::math::pow<2>( std::max( sig - mub, 0.0 ) ) / ( 2.0 * varSubS );
  }
  else
  {
          return boost::math::pow<2>( std::min( sig - muw, 0.0 ) ) / ( 2.0 * varSubS );
  }
}

/**
 * Get a point transformed by an homography
 * @param xi
 * @param yi
 * @param mH
 */
inline cctag::Point2dN<double> getHPoint(
        const double xi,
        const double yi,
        const cctag::numerical::BoundedMatrix3x3d & mH )
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;
  BoundedVector3d vh;
  vh( 0 ) = xi; vh( 1 ) = yi; vh( 2 ) = 1.0;

  return cctag::Point2dN<double>( prec_prod<BoundedVector3d>( mH, vh ) );
}

bool orazioDistance(
        IdSet& idSet,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds);

bool orazioDistanceRobust(
        std::vector<std::list<double> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds);


/**
 * @brief (rectifSignal) Extract 1D signal from an image. Pick up signal from image of points on line [0;0] [0;1] in plan defined by \p mH.
 * 
 * @param[out] rectifSig rectified signal
 * @param[in] sView source grayscale image
 * @param[in] mH homographie qui permet de passer du plan du support Ã  l'image.
 * @param[in] n signal length
 * @param begin ?
 * @param end ?
 */

void extractSignalUsingHomography(
        cctag::ImageCut & rectifiedSig,
        const cv::Mat & src,
        cctag::numerical::BoundedMatrix3x3d & mH,
        const std::size_t n = 100,
        const double begin = 0.0,
        const double end = 1.0 );

/**
 * @brief Collect cuts from a 2D line in the image (line is from pStart to pStop)
 * 
 * @param[out] cut signal collected from pStart to pStop (\p n values).
 * @param[in] sView source image
 * @param[in] pStart 2D position in the image coordinates
 * @param[in] pStop 2D position in the image coordinates
 * @param[in] n number of steps
 */

std::size_t cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::Point2dN<double> & pStart,
        const cctag::Point2dN<double> & pStop,
        const std::size_t nSteps );

/**
 * Collect signal from center to external ellipse point
 * @param[out] cuts
 * @param[in] sourceView
 * @param[in] center point from where all cuts starts from
 * @param[in] pts points on an external ellipse
 */

void collectCuts(
        std::vector<cctag::ImageCut> & cuts, 
        const cv::Mat & src,
        const cctag::Point2dN<double> & center,
        const std::vector< cctag::Point2dN<double> > & pts,
        const std::size_t sampleCutLength,
        const std::size_t startOffset );


inline float getPixelBilinear(const cv::Mat & img, float x, float y)
{
  int px = (int)x; // floor of x
  int py = (int)y; // floor of y
  const uchar* p0 = img.data + px + py * img.step; // pointer to first pixel

  // load the four neighboring pixels
  const uchar & p1 = p0[0 + 0 * img.step];
  const uchar & p2 = p0[1 + 0 * img.step];
  const uchar & p3 = p0[0 + 1 * img.step];
  const uchar & p4 = p0[1 + 1 * img.step];

  // Calculate the weights for each pixel
  float fx = x - px;
  float fy = y - py;
  float fx1 = 1.0f - fx;
  float fy1 = 1.0f - fy;

  float w1 = fx1 * fy1;
  float w2 = fx  * fy1;
  float w3 = fx1 * fy;
  float w4 = fx  * fy;

  // Calculate the weighted sum of pixels (for each color channel)
  return (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4)/4.0f;
}

inline float getPixelBicubic(const cv::Mat & img, float vX, float vY)
//(IplImage *img, int newWidth, int newHeight)
{
      
//        IplImage *img2 = createImage(newWidth,newHeight);
//        img2 = createImage(newWidth,newHeight);
//        uchar * data = img->imageData;
//        uchar * Data = img2->imageData;
//        int a,b,c,d,index;
//        uchar Ca,Cb,Cc;
//        uchar C[5];
//        uchar d0,d2,d3,a0,a1,a2,a3;
//        int i,j,k,ii,jj;
//        int x,y;
//        float dx,dy;
//        float tx,ty;
//        
//        tx = (float)img->width /newWidth ;
//        ty =  (float)img->height / newHeight;
//        printf("%d %d", newWidth, newHeight);
//        
//        for(i=0; i<newHeight; i++)
//        for(j=0; j<newWidth; j++)
//        {
                  // printf("%d : %d\n",i,j);
           //x = (int)(tx*j);
           //y =(int)(ty*i);
           
  uchar C[5];
  uchar d0,d2,d3,a0,a1,a2,a3;

  
  const int x = (int) vX;
  const int y = (int) vY;

  const float dx = vX - x;
  const float dy = vY - y;

  //dx= tx*j-x;
  //dy=ty*i -y;
           
  for(int jj=0;jj<=3;jj++)
  {
     d0 = img.data[(y-1+jj)*img.step + (x-1)] - img.data[(y-1+jj)*img.step + (x) ] ;
     d2 = img.data[(y-1+jj)*img.step + (x+1)] - img.data[(y-1+jj)*img.step + (x) ] ;
     d3 = img.data[(y-1+jj)*img.step + (x+2)] - img.data[(y-1+jj)*img.step + (x) ] ;
     a0 = img.data[(y-1+jj)*img.step + (x)];
     a1 =  -1.0/3*d0 + d2 -1.0/6*d3;
     a2 = 1.0/2*d0 + 1.0/2*d2;
     a3 = -1.0/6*d0 - 1.0/2*d2 + 1.0/6*d3;
     C[jj] = a0 + a1*dx + a2*dx*dx + a3*dx*dx*dx;
  }
              
  d0 = C[0]-C[1];
  d2 = C[2]-C[1];
  d3 = C[3]-C[1];
  a0=C[1];
  a1 =  -1.0/3*d0 + d2 -1.0/6*d3;
  a2 = 1.0/2*d0 + 1.0/2*d2;
  a3 = -1.0/6*d0 - 1.0/2*d2 + 1.0/6*d3;
  return a0 + a1*dy + a2*dy*dy + a3*dy*dy*dy;
                 
                // if((int)Cc>255) Cc=255;
//                 if((int)Cc<0) Cc=0;
                 //Data[i*img2->widthStep +j*img2->nChannels +k ] = Cc;  
}


double costSelectCutFun(
        const std::vector<double> & varCuts,
        const boost::numeric::ublas::vector<std::size_t> & randomIdx,
        const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const double alpha = 10 );

/**
 * @brief Select the image cut with the higgest variance.
 *
 * @param[out] selection result of the selection of best cuts from \p collectedCuts
 * @param[in] selectSize number of desired cuts in \p selection
 * @param[in] collectedCuts all collected cuts
 * @param[in] sourceView
 * @param[in] dx gradient x
 * @param[in] dy gradient y
 * @param[in] refinedSegSize size of the segment used to refine the external point of cut
 *
 * @todo in progress
 */

void selectCut(
      std::vector< cctag::ImageCut > & cutSelection,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize,
        const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const double refinedSegSize,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials );

void selectCutNaive(
        std::vector< cctag::ImageCut > & cutSelection,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize, const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy );

/**
 * 	
 * @param mH
 * @param mEllipse
 * @param o
 * @param p
 * @return 
 */
cctag::numerical::BoundedMatrix3x3d adjustH(
        cctag::numerical::BoundedMatrix3x3d & mH,
	const cctag::Point2dN<double> & o,
	const cctag::Point2dN<double> & p );

/**
 * @brief Get signal
 *
 * @param[out] mH matrix
 * @param[out] signals
 * @param[in] lengthSig
 * @param[in] o
 * @param[in] vecExtPoint
 * @param[in] sourceView
 * @param[in] matEllipse
 */

bool getSignals(
        cctag::numerical::BoundedMatrix3x3d & mH,
        std::vector< cctag::ImageCut > & signals,
        const std::size_t lengthSig,
        const cctag::Point2dN<double> & o,
        const std::vector< cctag::Point2dN<double> > & vecExtPoint,
        const cv::Mat & src,
        const cctag::numerical::BoundedMatrix3x3d & matEllipse );

bool refineConicFamily(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & fsig,
        const std::size_t lengthSig,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2dN<double> > & pr,
        const bool useLmDif );

/**
 * @brief Get signal
 *
 * @param[out] cctag cctag to optimize/wrap
 * @param[out] fsig signal along the cuts at the end of the optimization
 * @param[in] lengthSig
 * @param[in] src source image
 * @param[in] ellipse outer ellipse (todo: is that already in the cctag object?)
 * @param[in] pr (todo: no clue)
 */
bool refineConicFamilyNew(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & fsig,
        const std::size_t lengthSig,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2dN<double> > & pr);

double imageCenterOptimizationNew(
        cctag::numerical::BoundedMatrix3x3d & mH,
        std::vector< cctag::ImageCut > & signals,
        cctag::Point2dN<double> & center,
        const double neighbourSize,
        const std::size_t lengthSig,
        const std::vector< cctag::Point2dN<double> > & vecExtPoint, 
        const cv::Mat & src, 
        const cctag::numerical::geometry::Ellipse & ellipse);

void getNearbyPoints(
          const cctag::numerical::geometry::Ellipse & ellipse,
          const cctag::Point2dN<double> & center,
          std::vector<cctag::Point2dN<double> > & nearbyPoints,
          const double neighbourSize,
          const NeighborType neighborType);

double costFunctionNew( cctag::numerical::BoundedMatrix3x3d & mH,
        std::vector< cctag::ImageCut > & signals,
        const std::size_t lengthSig,
        const cctag::Point2dN<double> & o,
        const std::vector< cctag::Point2dN<double> > & vecExtPoint, 
        const cv::Mat & src, 
        const cctag::numerical::BoundedMatrix3x3d & matEllipse );

/**
 * Identify a marker (robust way)
 *
 * @param[out] id
 * @param[out] markerHomography
 * @param[out] centerPoint
 * @param[in] ellipse outer ellipse of the marker
 * @param[in] ellipsePoints
 * @param[in] radiusRatios
 * @param[in] sourceView
 * @param[in] dx
 * @param[in] dy
 * @return detected as a marker ?
 */
int identify(
	CCTag & cctag,
	const std::vector< std::vector<double> > & radiusRatios, ///@todo directly use the bank
	const cv::Mat & src,
	const cv::Mat & dx,
	const cv::Mat & dy,
	const cctag::Parameters & params);
} // namespace cctag

#endif
