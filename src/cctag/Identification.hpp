#pragma once

#include <cctag/VisualDebug.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/algebra/matrix/Operation.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/algebra/svd.hpp>
#include <cctag/statistic/statistic.hpp>
#include <cctag/boostCv/cvImage.hpp>

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

//#define NAIVE_SELECTCUT

namespace popart {
class TagPipe;
class NearbyPoint;
};

namespace cctag {
namespace identification {

enum NeighborType {
  GRID,
  CIRCULAR,
  COV
};
  
/**
 * @brief Identify a marker:
 *   i) its imaged center is optimized: A. 1D image cuts are selected ; B. the optimization is performed 
 *   ii) the outer ellipse + the obtained imaged center delivers the image->cctag homography
 *   iii) the rectified 1D signals are read and deliver the ID via a nearest neighbour
 *        approach where the distance to the cctag bank's profiles used is the one described in [Orazio et al. 2011]
 * @param[in] tagIndex a sequence number assigned to this tag
 * @param[in] cctag whose center is to be optimized in conjunction with its associated homography.
 * @param[out] vSelectedCuts step 1 does nothing else than create cuts for this tag
 * @param[in] radiusRatios bank of radius ratios along with their associated IDs.
 * @param[in] src original gray scale image (original scale, uchar)
 * @param[in] params set of parameters
 * @return status of the markers (c.f. all the possible status are located in CCTag.hpp) 
 */
int identify_step_1(
    const int tagIndex,
	const CCTag & cctag,
    std::vector<cctag::ImageCut>& vSelectedCuts,
	// const std::vector< std::vector<double> > & radiusRatios,
	const cv::Mat & src,
    // popart::TagPipe* pipe,
	const cctag::Parameters & params);

/**
 * @brief Identify a marker:
 *   i) its imaged center is optimized: A. 1D image cuts are selected ; B. the optimization is performed 
 *   ii) the outer ellipse + the obtained imaged center delivers the image->cctag homography
 *   iii) the rectified 1D signals are read and deliver the ID via a nearest neighbour
 *        approach where the distance to the cctag bank's profiles used is the one described in [Orazio et al. 2011]
 * @param[in] tagIndex a sequence number assigned to this tag
 * @param[in] cctag whose center is to be optimized in conjunction with its associated homography.
 * @param[in] vSelectedCuts pre-generated cuts
 * @param[in] radiusRatios bank of radius ratios along with their associated IDs.
 * @param[in] src original gray scale image (original scale, uchar)
 * @param[in] params set of parameters
 * @return status of the markers (c.f. all the possible status are located in CCTag.hpp) 
 */
int identify_step_2(
    const int tagIndex,
	CCTag & cctag,
    std::vector<cctag::ImageCut>& vSelectedCuts,
	const std::vector< std::vector<double> > & radiusRatios,
	const cv::Mat & src,
    popart::TagPipe* pipe,
	const cctag::Parameters & params);

typedef std::vector< std::vector<double> > RadiusRatioBank;
typedef std::vector< std::pair< cctag::Point2dN<double>, cctag::ImageCut > > CutSelectionVec;

/**
 * @brief Apply a planar homography to a 2D point.
 * 
 * @param[out] xRes x coordinate of the transformed point
 * @param[out] yRes y coordinate of the transformed point
 * @param[in] mHomography the transformation represented by a 3x3 matrix
 * @param[in] x coordinate of the point to be transformed
 * @param[in] y coordinate of the point to be transformed
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

/**
 * @brief Read and identify a 1D rectified image signal.
 * 
 * @param[out] vScore ordered set of the probability of the k nearest IDs
 * @param[in] rrBank Set of vector of radius ratio describing the 1D profile of the cctag library
 * @param[in] cuts image cuts holding the rectified 1D signal
 * @param[in] minIdentProba minimal probability to considered a cctag as correctly identified
 * @return true if the cctag has been correctly identified, false otherwise
 */
// todo: change the data structure here: map for id+scores, thresholding with minIdentProba
// outside of this function
bool orazioDistanceRobust(
        std::vector<std::list<double> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const double minIdentProba);

/**
 * @brief Extract a rectified 1D signal along an image cut based on an homography.
 * 
 * @param[out] cut image cut holding the rectified image signal
 * @param[in] src source grayscale image (uchar)
 * @param[in] mHomography image->cctag homography
 * @param[in] mInvHomography cctag>image homography
 */
void extractSignalUsingHomography(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        const cctag::numerical::BoundedMatrix3x3d & mInvHomography);

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
 * @param[out] cut image cut that will hold the 1D image signal regularly 
 *             collected from cut.beginSig() to cut.endSig()
 * @param[in] src source gray scale image (uchar)
 */
void cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src);

/**
 * @brief Collect signals (image cuts) from center to outer ellipse points
 * 
 * @param[out] cuts collected cuts
 * @param[in] src source gray scale image (uchar)
 * @param[in] center outer ellipse center
 * @param[in] outerPoints outer ellipse points
 * @param[in] nSamplesInCut number of samples collected in an image cut
 * @param[in] beginSig offset from which the signal must be collected (in [0 1])
 */
void collectCuts(
        std::vector<cctag::ImageCut> & cuts, 
        const cv::Mat & src,
        const cctag::Point2dN<double> & center,
        const std::vector< cctag::DirectedPoint2d<double> > & outerPoints,
        const std::size_t sampleCutLength,
        const std::size_t startOffset );

/*
 * @brief Bilinear interpolation for a point whose coordinates are (x,y)
 * 
 * @param[in] src source gray scale image (uchar)
 * @param[in] x x coordinate
 * @param[in] y y coordinate
 * @return computed pixel value
 */
inline float getPixelBilinear(const cv::Mat & src, float x, float y)
{
  int px = (int)x; // floor of x
  int py = (int)y; // floor of y
  const uchar* p0 = src.data + px + py * src.step; // pointer to first pixel
  
  // load the four neighboring pixels
  const uchar & p1 = p0[0 + 0 * src.step];
  const uchar & p2 = p0[1 + 0 * src.step];
  const uchar & p3 = p0[0 + 1 * src.step];
  const uchar & p4 = p0[1 + 1 * src.step];

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
  return (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4)/2; // todo: was initially /4. Justification: make sense for a storage back in uchar to avoid overflow ??
}

/**
 * @brief Compute a cost (score) for the cut selection based on the variance and the gradient orientation
 * of the outer point (cut.stop()) over all outer points.
 * 
 * @param[in] varCuts variance of the signal over all image cuts
 * @param[in] outerPoints cut.stop(), i.e. outer ellipse point, stop points of the considered image cuts
 * @param[in] randomIdx random index subsample in [0 , outerPoints.size()[
 * @param[in] alpha magic hyper parameter
 * @return score
 */
double costSelectCutFun(
        const std::vector<double> & varCuts,
        const std::vector< cctag::DirectedPoint2d<double> > & outerPoints,
        const boost::numeric::ublas::vector<std::size_t> & randomIdx,
        const double alpha = 10 );

/**
 * @brief Select a subset of image cuts appropriate for the image center optimisation.
 * This selection aims at maximizing the variance of the image signal over all the 
 * selected cuts while minimizing the norm of the sum of the normalized gradient over
 * all outer points, i.e. all cut.stop().
 *
 * @param[out] vSelectedCuts selected image cuts
 * @param[in] selectSize number of desired cuts to select
 * @param[in] collectedCuts all the collected cuts
 * @param[in] src source gray scale image (uchar)
 * @param[in] refinedSegSize deprec (do not remove)
 * @param[in] cutsSelectionTrials number of random draw
 */
void selectCut(
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const double refinedSegSize,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials );

void selectCutCheap(
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const double refinedSegSize,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials );

#ifdef NAIVE_SELECTCUT
void selectCutNaive( // depreciated: dx and dy are not accessible anymore -> use DirectedPoint instead
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize, const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy );
#endif // NAIVE_SELECTCUT

/**
 * @brief Collect rectified 1D signals along image cuts.
 * 
 * @param[out] vCuts set of the image cuts whose the rectified signal is to be to computed
 * @param[in] mHomography transformation image->cctag used to rectified the 1D signal
 * @param[in] src source gray scale image (uchar)
 */
void getSignals(
        std::vector< cctag::ImageCut > & vCuts,
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        const cv::Mat & src);

/**
 * @brief Compute the optimal homography/imaged center based on the 
 * signal in the image and  the outer ellipse, supposed to be image the unit circle.
 * 
 * @param[in] tagIndex a sequence number for this tag
 * @param[out] mHomography image->cctag homography to optimize
 * @param[out] optimalPoint imaged center to optimize
 * @param[out] vCuts cuts holding the rectified 1D signals at the end of the optimization
 * @param[in] src source image
 * @param[in] ellipse outer ellipse (todo: is that already in the cctag object?)
 * @param[in] params parameters of the cctag algorithm
 * @return true if the optimization has found a solution, false otherwise.
 */
bool refineConicFamilyGlob(
        const int tagIndex,
        cctag::numerical::BoundedMatrix3x3d & mHomography,
        Point2dN<double> & optimalPoint,
        std::vector< cctag::ImageCut > & vCuts, 
        const cv::Mat & src,
        popart::TagPipe* cudaPipe,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters params,
        popart::NearbyPoint* cctag_pointer_buffer );

/**
 * @brief Convex optimization of the imaged center within a point's neighbourhood.
 * 
 * @param[out] mHomography optimal homography from the pixel plane to the cctag plane.
 * @param[out] vCuts vector of the image cuts whose the signal has been rectified w.r.t. the computed mHomography
 * @param[out] center optimal imaged center
 * @param[out] minRes residual after optimization
 * @param[in] neighbourSize size of the neighbourhood to consider relatively to the outer ellipse dimensions
 * @param[in] gridNSample number of sample points along one dimension of the neighbourhood (e.g. grid)
 * @param[in] src source gray (uchar) image
 * @param[inout] cudaPipe CUDA object handle, changing
 * @param[in] outerEllipse outer ellipse
 * @param[in] params Parameters read from config file
 */
bool imageCenterOptimizationGlob(
        cctag::numerical::BoundedMatrix3x3d & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        cctag::Point2dN<double> & center,
        double & minRes,
        const double neighbourSize,
        const cv::Mat & src, 
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters params );


/**
 * @brief Compute a set of point locations nearby a given center following
 * a given type of pattern (e.g. regularly sampled points over a grid)
 * 
 * @param[in] ellipse outer ellipse providing the absolute scale.
 * @param[in] center the center around which point locations are computed
 * @param[out] nearbyPoints computed 2D points
 * @param[in] neighbourSize provide the scale of the pattern relatively to the ellipse dimensions
 * @param[in] gridNSample number of sampled points along a dimension of the pattern
 * @param[in] neighborType type of pattern (e.g. a grid)
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
 * @param[in] ellipse ellipse image of the unit circle
 * @param[in] center imaged center, projection of the origin
 * @param[out] mHomography computed homography
 */
void computeHomographyFromEllipseAndImagedCenter(
        const cctag::numerical::geometry::Ellipse & ellipse,
        const cctag::Point2dN<double> & center,
        cctag::numerical::BoundedMatrix3x3d & mHomography);

/**
 * @brief Compute the residual of the optimization which is the average of the square of the 
 * differences between two rectified image signals/cuts over all possible cut-pair in a set 
 * of image cuts.
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


/**
 * @brief COmpute a median value from a vector of scalar values
 * 
 * @param[vec] vec vector of scalar values
 * @return its associated median
 */
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

