/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/utils/VisualDebug.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/Statistic.hpp>

#include <opencv2/opencv.hpp>

#include <boost/foreach.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <cmath>
#include <vector>

//#define NAIVE_SELECTCUT

namespace cctag {
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
    int tagIndex,
	const CCTag & cctag,
    std::vector<cctag::ImageCut>& vSelectedCuts,
	// const std::vector< std::vector<float> > & radiusRatios,
	const cv::Mat & src,
    // cctag::TagPipe* pipe,
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
    int tagIndex,
	CCTag & cctag,
    std::vector<cctag::ImageCut>& vSelectedCuts,
	const std::vector< std::vector<float> > & radiusRatios,
	const cv::Mat & src,
    cctag::TagPipe* cudaPipe,
	const cctag::Parameters & params);

using RadiusRatioBank = std::vector<std::vector<float>>;
using CutSelectionVec =  std::vector< std::pair< cctag::Point2d<Eigen::Vector3f>, cctag::ImageCut>>;

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
        float & xRes,
        float & yRes,
        const Eigen::Matrix3f & mHomography,
        const float x,
        const float y)
{
  float u = mHomography(0,0)*x + mHomography(0,1)*y + mHomography(0,2);
  float v = mHomography(1,0)*x + mHomography(1,1)*y + mHomography(1,2);
  float w = mHomography(2,0)*x + mHomography(2,1)*y + mHomography(2,2);
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
bool orazioDistanceRobust(
        std::vector<std::list<float> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        float minIdentProba);

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
        const Eigen::Matrix3f & mHomography,
        const Eigen::Matrix3f & mInvHomography);

/* deprecated */
void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedCut,
        const cv::Mat & src,
        Eigen::Matrix3f & mHomography,
        std::size_t nSamples = 100,
        float begin = 0.f,
        float end = 1.f );

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

std::pair<float,float> convImageCut(const std::vector<float> & kernel, ImageCut & cut);

void blurImageCut(float sigma, cctag::ImageCut & cut);

bool outerEdgeRefinement(ImageCut & cut, const cv::Mat & src, float scale, size_t numSamplesOuterEdgePointsRefinement);

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
        const cctag::Point2d<Eigen::Vector3f> & center,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & outerPoints,
        std::size_t sampleCutLength,
        std::size_t startOffset );

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
  return (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4)/2;
}

/**
 * @brief Collect rectified 1D signals along image cuts.
 * 
 * @param[out] vCuts set of the image cuts whose the rectified signal is to be to computed
 * @param[in] mHomography transformation image->cctag used to rectified the 1D signal
 * @param[in] src source gray scale image (uchar)
 */
void getSignals(
        std::vector< cctag::ImageCut > & vCuts,
        const Eigen::Matrix3f & mHomography,
        const cv::Mat & src);

/**
 * @brief Compute the optimal homography/imaged center based on the 
 * signal in the image and  the outer ellipse, supposed to be image the unit circle.
 * 
 * @param[in] tagIndex a sequence number for this tag
 * @param[out] mHomography image->cctag homography to optimize
 * @param[out] optimalPoint imaged center to optimize
 * @param[out] vCuts cuts holding the rectified 1D signals at the end of the optimization
 * @param[out] residual from the optimization (normalized w.r.t. binary pattern)
 * @param[in] src source image
 * @param[in] outerEllipse outer ellipse
 * @param[in] params parameters of the cctag algorithm
 * @return true if the optimization has found a solution, false otherwise.
 */
bool refineConicFamilyGlob(
        int tagIndex,
        Eigen::Matrix3f & mHomography,
        Point2d<Eigen::Vector3f> & optimalPoint,
        std::vector< cctag::ImageCut > & vCuts, 
        const cv::Mat & src,
        cctag::TagPipe* cudaPipe,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters & params,
        cctag::NearbyPoint* cctag_pointer_buffer,
        float & residual);

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
        Eigen::Matrix3f & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        cctag::Point2d<Eigen::Vector3f> & center,
        float & minRes,
        float neighbourSize,
        const cv::Mat & src, 
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters & params );


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
          const cctag::Point2d<Eigen::Vector3f> & center,
          std::vector<cctag::Point2d<Eigen::Vector3f> > & nearbyPoints,
          float neighbourSize,
          std::size_t gridNSample,
          NeighborType neighborType);

/**
 * @brief Compute an homography (up to a 2D rotation) based on its imaged origin [0,0,1]'
 * and its imaged unit circle (represented as an ellipse, assuming only quasi-affine transformation
 * 
 * @param[in] ellipse ellipse image of the unit circle
 * @param[in] center imaged center, projection of the origin
 * @param[out] mHomography computed homography
 */
void computeHomographyFromEllipseAndImagedCenter(
        const cctag::numerical::geometry::Ellipse & ellipse,
        const cctag::Point2d<Eigen::Vector3f> & center,
        Eigen::Matrix3f & mHomography);

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
float costFunctionGlob(
        const Eigen::Matrix3f & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        const cv::Mat & src,
        bool & flag);


/**
 * @brief COmpute a median value from a vector of scalar values
 * 
 * @param[vec] vec vector of scalar values
 * @return its associated median
 * @todo PERFORMANCE! take a pair of iterators, will avoid copying at the call site.
 */
template<typename VecT>
typename VecT::value_type computeMedian( const VecT& vec )
{
  using T = typename VecT::value_type;
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

inline float dis( const float sig, const float val, const float mub, const float muw, const float varSubS )
{
  if ( val == -1 )
  {
          return boost::math::pow<2>( std::max( sig - mub, 0.f ) ) / ( 2.0f * varSubS );
  }
  else
  {
          return boost::math::pow<2>( std::min( sig - muw, 0.f ) ) / ( 2.0f * varSubS );
  }
}

/* depreciated */
bool refineConicFamily(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & fsig,
        std::size_t lengthSig,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2d<Eigen::Vector3f> > & pr,
        bool useLmDif );

/* depreciated */
/**
 * Get a point transformed by an homography
 * @param xi
 * @param yi
 * @param mH
 */
inline cctag::Point2d<Eigen::Vector3f> getHPoint(
        const float xi,
        const float yi,
        const Eigen::Matrix3f& mH )
{
  using namespace cctag::numerical;
  Eigen::Vector3f vh;
  vh( 0 ) = xi; vh( 1 ) = yi; vh( 2 ) = 1.f;

  return cctag::Point2d<Eigen::Vector3f>(mH * vh);
}

/* depreciated */
bool orazioDistance(
        IdSet& idSet,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        std::size_t startOffset,
        float minIdentProba,
        std::size_t sizeIds);

} // namespace identification
} // namespace cctag

