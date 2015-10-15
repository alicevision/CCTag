#ifndef _CCTAG_CCTAG_IDENTIFICATION_HPP_
#define	_CCTAG_CCTAG_IDENTIFICATION_HPP_

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
namespace identification {

enum NeighborType {
  GRID,
  CIRCULAR,
  COV
};
  
/**
 * Identify a marker: i) its imaged center is optimized 
 *                    ii) the outer ellipse + the obtained imaged center deliver the image->cctag homography
 *                    iii) the rectified 1D signal(s) is(are) read and deliver the ID via a nearest neighbour 
 *                    approach where the metric used is the one described in Orazio et al. 2011
 * @param[in] cctag whose center is to be optimized in conjunction with its associated homography.
 * @param[in] radiusRatios bank of radius ratios along with their associated IDs.
 * @param[in] src original image (original scale)
 * @param[in] params set of parameters
 * @return status of the markers (c.f. all the possible status are located in CCTag.hpp) 
 */
int identify(
	CCTag & cctag,
	const std::vector< std::vector<double> > & radiusRatios,
	const cv::Mat & src,
	const cctag::Parameters & params);

typedef std::vector< std::vector<double> > RadiusRatioBank;
typedef std::vector< std::pair< cctag::Point2dN<double>, cctag::ImageCut > > CutSelectionVec;

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

/**
 * Apply an planar homography to a 2D point.
 * @param[out] xRes x coordinate of the transformed point
 * @param[out] yRes y coordinate of the transformed point
 * @param[in] mHomography the transformation represented by a 3x3 matrix
 * @param[in] x coordinate of the point
 * @param[in] y coordinate of the point
 */
inline void applyHomography(
        double & xRes,
        double & yRes,
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        const double x,
        const double y)
{
  double u = mHomography(0,0)*x + mHomography(0,1)*y + mHomography(0,2);
  double v = mHomography(1,0)*x + mHomography(1,1)*y + mHomography(1,2);
  double w = mHomography(2,0)*x + mHomography(2,1)*y + mHomography(2,2);
  xRes = u/w;
  yRes = v/w;
}

bool orazioDistanceRobust(
        std::vector<std::list<double> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds);


/**
 * @brief Extract a rectified 1D signal along an image cut based on an homography.
 * 
 * @param[out] cut image cut holding the rectified image signal
 * @param[in] src source grayscale (uchar) image
 * @param[in] mHomography image->cctag homography
 */
void extractSignalUsingHomography(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::numerical::BoundedMatrix3x3d & mHomography);

/* depreciated */
void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedSig,
        const cv::Mat & src,
        cctag::numerical::BoundedMatrix3x3d & mHomography,
        std::size_t nSamples = 100,
        const double begin = 0.0,
        const double end = 1.0 );

/**
 * @brief Extract a regularly sampled 1D signal along an image cut.
 * 
 * @param[out] cut image cut holding the 1D image signal collected
 * @param[in] src source grayscale (uchar) image
 * @param[in] mHomography image->cctag homography
 * @param[in] pStart starting point of the cut
 * @param[in] pStop stopping point of the cut
 * @param[in] nSamples number of sample along the image cut
 * @return ? todo
 */
std::size_t cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::Point2dN<double> & pStart,
        const cctag::DirectedPoint2d<double> & pStop,
        const std::size_t nSamples);

/**
 * Collect signals (image cuts) from center to outer ellipse points
 * 
 * @param[out] cuts collected cuts
 * @param[in] src source image (gray)
 * @param[in] center outer ellipse center
 * @param[in] outerPoints outer ellipse points
 * @param[in] sampleCutLength number of samples collected in an image cut
 * @param[in] startOffset in [0 sampleCutLength-1], represents the offset from which the signal must be available (from startOffset to sampleCutLength-1)
 */
void collectCuts(
        std::vector<cctag::ImageCut> & cuts, 
        const cv::Mat & src,
        const cctag::Point2dN<double> & center,
        const std::vector< cctag::DirectedPoint2d<double> > & outerPoints,
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

double costSelectCutFun(
        const std::vector<double> & varCuts,
        const std::vector< cctag::DirectedPoint2d<double> > & outerPoints,
        const boost::numeric::ublas::vector<std::size_t> & randomIdx,
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
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const double refinedSegSize,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials );

void selectCutNaive( // depreciated: dx and dy are not accessible anymore -> use DirectedPoint instead
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize, const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy );

/**
 * 	
 * @param mHomography
 * @param mEllipse
 * @param center
 * @param point
 * @return 
 */
void centerScaleRotateHomography(
        cctag::numerical::BoundedMatrix3x3d & mHomography,
	const cctag::Point2dN<double> & center,
	const cctag::DirectedPoint2d<double> & point);

/**
 * @brief Collect and compute the rectified 1D signals along image cuts.
 * 
 * @param[out] vCuts vector of the image cuts whose the rectified signal is to be to computed
 * @param[in] mHomography transformation used to rectified the 1D signal from the pixel plane to the cctag plane.
 * @param[in] src source grayscale image (uchar)
 */
void getSignals(
        std::vector< cctag::ImageCut > & vCuts,
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        const cv::Mat & src);

/**
 * @brief Compute the optimal homography/imaged center based on the 
 * signal in the image, the cctag symmetry constraints and also the outer ellipse,
 * image the unit circle, which is supposed to be known.
 * 
 * @param[out] cctag cctag to optimize in order to find its imaged center in conjunction 
 * with the image->cctag homography
 * @param[out] vCuts cuts holding the rectified 1D signals at the end of the optimization
 * @param[in] nSamples number of samples on image cuts
 * @param[in] src source image
 * @param[in] ellipse outer ellipse (todo: is that already in the cctag object?)
 * @return true if the optimization has converged, false otherwise.
 */
bool refineConicFamilyGlob(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & vCuts,
        const std::size_t nSamples,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse);

/**
 * @brief Convex optimization of the imaged center within a point's neighbourhood.
 * 
 * @param[out] mHomography optimal homography from the pixel plane to the cctag plane.
 * @param[out] vCuts vector of the image cuts whose the signal has been rectified w.r.t. the computed mHomography
 * @param[out] center optimal imaged center
 * @param[out] neighbourSize obtained residual
 * @param[in] neighbourSize size of the neighbourhood to consider relatively to the outer ellipse dimensions
 * @param[in] gridNSample number of sample points along one dimension of the neighbourhood (e.g. grid)
 * @param[in] src source gray (uchar) image
 * @param[in] outerEllipse outer ellipse
 */
bool imageCenterOptimizationGlob(
        cctag::numerical::BoundedMatrix3x3d & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        cctag::Point2dN<double> & center,
        double & minRes,
        const double neighbourSize,
        const std::size_t gridNSample,
        const cv::Mat & src, 
        const cctag::numerical::geometry::Ellipse & outerEllipse);

/**
 * @brief Compute a set of point locations nearby a given center following
 * a given type of pattern (e.g. regularly sampled points)
 * 
 * @param[in] ellipse outer ellipse providing the absolute scale.
 * @param[in] center the center around which point locations are computed
 * @param[in] neighbourSize provide the scale of the pattern relatively to the ellipse dimensions
 * @param[in] gridNSample number of sampled points along a dimension of the pattern
 * @param[in] neighborType type of pattern (e.g. a grid)
 * @param[out] nearbyPoints computed 2D points
 */
void getNearbyPoints(
          const cctag::numerical::geometry::Ellipse & ellipse,
          const cctag::Point2dN<double> & center,
          std::vector<cctag::Point2dN<double> > & nearbyPoints,
          const double neighbourSize,
          const std::size_t gridNSample,
          const NeighborType neighborType);

/**
 * @brief Compute an homography (up to a 2D rotation) based on its imaged origin [0,0,1]'
 * and its imaged unit circle (represented as an ellipse, assuming only quasi-affine transformation
 * PS: this version will be replaced by its closed-form formulation (todo)
 * 
 * @param[in] mEllipse ellipse matrix, projection of the unit circle
 * @param[in] center imaged center, projection of the origin
 * @param[out] mHomography computed homography
 */
void computeHomographyFromEllipseAndImagedCenter(
        const cctag::numerical::BoundedMatrix3x3d & mEllipse,
        const cctag::Point2dN<double> & center,
        cctag::numerical::BoundedMatrix3x3d & mHomography);

/**
 * @brief Compute the residual of the optimization which is the average of the square of the 
 * differences between two rectified image signals/cuts over all possible cut-pair in the set 
 * of image cuts in vCuts.
 * 
 * @param[in] mHomography transformation used to rectified the 1D signal from the pixel plane to the cctag plane.
 * @param[out] vCuts vector of the image cuts holding the rectified signal according to mHomography
 * @param[in] src source gray scale image (uchar)
 * @param[out] flag: true if at least one image cut has been readable (within the image bounds), false otherwise.
 * @return residual
 */
double costFunctionGlob(
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        const cv::Mat & src,
        bool & flag);


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

/* depreciated */
bool refineConicFamily(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & fsig,
        const std::size_t lengthSig,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2dN<double> > & pr,
        const bool useLmDif );

/* depreciated */
bool orazioDistance(
        IdSet& idSet,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds);

} // namespace identification
} // namespace cctag

#endif
