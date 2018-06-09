/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Identification.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/2DTransform.hpp>

#undef SUBPIX_EDGE_OPTIM
#include <cctag/SubPixEdgeOptimizer.hpp>

#include <cctag/geometry/Circle.hpp>
#include <cctag/utils/Talk.hpp>

#ifdef WITH_CUDA
#include "cctag/cuda/tag.h"
#endif

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/assert.hpp>

#include <cmath>
#include <vector>

#include <tbb/tbb.h>

#include "cctag/cuda/onoff.h"

namespace cctag {
namespace identification {

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
        float minIdentProba)
{
  BOOST_ASSERT( cuts.size() > 0 );

  using namespace cctag::numerical;
  using namespace boost::accumulators;

  typedef std::map<float, MarkerID> MapT;

  if ( cuts.size() == 0 )
  {
    return false;
  }
#ifdef GRIFF_DEBUG
  if( rrBank.size() == 0 )
  {
    return false;
  }
#endif // GRIFF_DEBUG
  
  const size_t cut_count = cuts.size();
  static tbb::mutex vscore_mutex;

  tbb::parallel_for(size_t(0), cut_count, [&](size_t i) {
    const cctag::ImageCut& cut = cuts[i];
    if ( !cut.outOfBounds() )
    {
      MapT sortedId; // 6-nearest neighbours along with their affectation probability
      const std::size_t sizeIds = 6;
      IdSet idSet;
      idSet.reserve(sizeIds);

      // imgSig contains the rectified 1D signal.
      //boost::numeric::ublas::vector<float> imgSig( cuts.front().imgSignal().size() );
      const std::vector<float> & imgSig = cut.imgSignal();

      // compute some statitics
      accumulator_set< float, features< /*tag::median,*/ tag::variance > > acc;
      // Put the image signal into the accumulator
      acc = std::for_each( imgSig.begin()+30, imgSig.end(), acc ); // todo@Lilian +30

      // Mean
      const float medianSig = boost::accumulators::mean( acc );
      
      // or median
      //const float medianSig = computeMedian( imgSig );

      const float varSig = boost::accumulators::variance( acc );

      accumulator_set< float, features< tag::mean > > accInf;
      accumulator_set< float, features< tag::mean > > accSup;
      
      bool doAccumulate = false;
      for(float i : imgSig)
      {
        if ( (!doAccumulate) && ( i < medianSig ) )
          doAccumulate = true;
          
        if (doAccumulate)
        {
          if ( i < medianSig )
            accInf( i );
          else
            accSup( i );
        }
      }
      const float muw = boost::accumulators::mean( accSup );
      const float mub = boost::accumulators::mean( accInf );

      // Find the nearest ID in rrBank
      const float stepX = (cut.endSig() - cut.beginSig()) / ( imgSig.size() - 1.f );

      // vector of 1 or -1 values
      std::vector<float> digit( imgSig.size() );

  #ifdef GRIFF_DEBUG
      assert( rrBank.size() > 0 );
  #endif // GRIFF_DEBUG
      // Loop over imgSig values, compute and sum the difference between 
      // imgSig and digit (i.e. generated profile)
      for( std::size_t idc = 0; idc < rrBank.size(); ++idc )
      {
        // Compute the idc-th profile from the radius ratio
        // todo@Lilian: to be pre-computed
        float x = cut.beginSig();
        for(float & i : digit)
        {
          std::ssize_t ldum = 0;
          for(float j : rrBank[idc])
          {
            if( 1.f / j <= x )
            {
              ++ldum;
            }
          }
          // set odd value to -1 and even value to 1
          i = - ( ldum % 2 ) * 2 + 1;
          
          x += stepX;
        }

        // compute distance to profile
        float distance = 0;
        for( std::size_t i = 0 ; i < imgSig.size() ; ++i )
        {
          distance += dis( imgSig[i], digit[i], mub, muw, varSig );
        }
        const float v = std::exp( -distance ); // todo: remove the exp()
        sortedId[v] = idc;
      }

  #ifdef GRIFF_DEBUG
      assert( sortedId.size() > 0 );
  #endif // GRIFF_DEBUG
      int k = 0;
      BOOST_REVERSE_FOREACH( const MapT::const_iterator::value_type & v, sortedId )
      {
        if( k >= sizeIds ) break;
        std::pair< MarkerID, float > markerId;
        markerId.first = v.second;
        markerId.second = v.first;
        idSet.push_back(markerId);
        ++k;
      }

  #ifdef GRIFF_DEBUG
      assert( idSet.size() > 0 );
      MarkerID _debug_m = idSet.front().first;
      assert( _debug_m > 0 );
      assert( vScore.size() > _debug_m );
  #endif // GRIFF_DEBUG

      {
        tbb::mutex::scoped_lock lock(vscore_mutex);
        vScore[idSet.front().first].push_back(idSet.front().second);
      }
    }
  });
  return true;
}

void createRectifiedCutImage(const std::vector<ImageCut> & vCuts, cv::Mat & output)
{
  output = cv::Mat(vCuts.size(), vCuts.front().imgSignal().size(), CV_8UC1);
  for(int i=0 ; i<vCuts.size() ; ++i)
  {
    const ImageCut & cut = vCuts[i];
    for(int j=0 ; j < cut.imgSignal().size() ; ++j)
    {
      output.at<uchar>(i,j) = (uchar) cut.imgSignal()[j];
    }
  }
}

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
        const Eigen::Matrix3f & mInvHomography)
{
  using namespace boost;
  using namespace cctag::numerical;
  
  float xStart, xStop, yStart, yStop;
  
  float backProjStopX, backProjStopY;
  applyHomography(backProjStopX, backProjStopY, mInvHomography, cut.stop().x(), cut.stop().y());
  
  // Check whether the signal to be collected start at 0.f and stop at 1.f
  if ( cut.beginSig() != 0.f)
  {
    xStart = backProjStopX * cut.beginSig();
    yStart = backProjStopY * cut.beginSig();
  }else
  {
    xStart = 0;
    yStart = 0;
  }
  if ( cut.endSig() != 1.f)
  {
    xStop = backProjStopX * cut.endSig();
    yStop = backProjStopY * cut.endSig();
  }else
  {
    xStop  = backProjStopX;
    yStop  = backProjStopY;
  }

  // Compute the steps stepX and stepY along x and y.
  const std::size_t nSamples = cut.imgSignal().size();
  const float stepX = ( xStop - xStart ) / ( nSamples - 1.f );
  const float stepY = ( yStop - yStart ) / ( nSamples - 1.f );

  float xRes, yRes;

  float x =  xStart;
  float y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    applyHomography(xRes, yRes, mHomography, x, y);

    if ( xRes >= 1.f && xRes <= src.cols-1 &&
         yRes >= 1.f && yRes <= src.rows-1 )
    {
      // Bilinear interpolation
      cut.imgSignal()[i] = getPixelBilinear( src, xRes, yRes);
    }
    else
    {
      cut.setOutOfBounds(true);
    }
    
    x += stepX;
    y += stepY;
  }
  //const float sigma = 1.f;
  //blurImageCut(sigma, cut);
}

void blurImageCut(float sigma, std::vector<float> & signal)
{
  //const std::vector<float> kernel = { 0.0044f, 0.0540f, 0.2420f, 0.3991f, 0.2420f, 0.0540f, 0.0044f };
  const std::vector<float> kernel = { 0.0276f, 0.0663f, 0.1238f, 0.1802f, 0.2042f, 0.1802f, 0.1238f, 0.0663f, 0.0276f };
  
  std::vector<float> output;
  std::size_t sizeCut = signal.size();
  std::size_t sizeKernel = kernel.size();
  output.resize(sizeCut);
  
  std::size_t halfSize = (sizeKernel-1)/2;
  
  /*std::cout << "before = [" << std::endl;
  for ( std::size_t i=0 ; i<sizeCut; ++i)
  {
    std::cout << signal[i] << " , " ;
  }
  std::cout << "]; " << std::endl;*/
  
  for ( std::ssize_t i=0 ; i<sizeCut; ++i)
  {
    float tmp = 0;
    for ( std::size_t j=0 ; j<sizeKernel; ++j)
    {
      if ( ssize_t(i-halfSize+j) < 0 )
        tmp += signal[0]*kernel[j];
      else if( (i-halfSize+j) >=  sizeCut)
        tmp += signal[sizeCut-1]*kernel[j];
      else
        tmp += signal[i-halfSize+j]*kernel[j];
    }
    output[i] = tmp;
  }
  
//  std::cout << "convolved = [" << std::endl;
//  for ( std::size_t i=0 ; i<sizeCut; ++i)
//  {
//    std::cout << output[i] << " , " ;
//  }
//  std::cout << "]; " << std::endl;
  
  signal = output;
}

/* depreciated */
void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedCut,
        const cv::Mat & src,
        Eigen::Matrix3f & mHomography,
        std::size_t nSamples,
        float begin,
        float end)
{
  using namespace boost;
  using namespace cctag::numerical;

  BOOST_ASSERT( rectifiedCut.imgSignal().size() == 0 );
  BOOST_ASSERT( end >= begin );
  
  // Check wheter the image signal size has been properly allocated.
  BOOST_ASSERT( nSamples == rectifiedCut.imgSignal().size() );

  nSamples = rectifiedCut.imgSignal().size();
  
  const float stepXi = ( end - begin ) / ( nSamples - 1.f );

  rectifiedCut.start() = getHPoint( begin, 0.f, mHomography );
  rectifiedCut.stop() = cctag::DirectedPoint2d<Eigen::Vector3f>( getHPoint( end, 0.f, mHomography ), 0.f, 0.f);

  std::vector<std::size_t> idxNotInBounds;
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    const float xi = i * stepXi + begin;
    const cctag::Point2d<Eigen::Vector3f> hp = getHPoint( xi, 0.f, mHomography );

    if ( hp.x() >= 1.f && hp.x() <= src.cols-1 &&
         hp.y() >= 1.f && hp.y() <= src.rows-1 )
    {
      // Bilinear interpolation
      rectifiedCut.imgSignal()[i] = getPixelBilinear( src, hp.x(), hp.y());
    }
    else
    {
      rectifiedCut.setOutOfBounds(true);
    }
  }
}

/// Code usable to apply a 1D gaussian filtering at the end of the extractSignalUsingHomography
///
//  float guassOneD[] = { 0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044 };
//  for( std::size_t i = 0; i < nSamples; ++i )
//  {
//    float tmp = 0;
//    for ( std::size_t j=0 ; j<7; ++j)
//    {
//      if ( (i-3+j >= 0) && (i-3+j < nSamples) )
//      {
//        tmp += rectifiedCut.imgSignal()(i-3+j)*guassOneD[j];
//      }
//    }
//    rectifiedCut.imgSignal()(i) = tmp;
//  }
///


/**
 * @brief Extract a regularly sampled 1D signal along an image cut.
 * 
 * @param[out] cut image cut that will hold the 1D image signal regularly 
 *             collected from cut.beginSig() to cut.endSig()
 * @param[in] src source gray scale image (uchar)
 */
void cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src)
{
  float xStart, yStart, xStop, yStop;
  const float diffX = cut.stop().x() - cut.start().x();
  const float diffY = cut.stop().y() - cut.start().y();
  
  // Check whether the signal to be collected start at 0.f and stop at 1.f
  if ( cut.beginSig() != 0.f)
  {
    // Don't start at the beginning of the cut.
    xStart = cut.start().x() + diffX * cut.beginSig();
    yStart = cut.start().y() + diffY * cut.beginSig();
  }else
  {
    xStart = cut.start().x();
    yStart = cut.start().y();
  }
  if ( cut.endSig() != 1.f)
  {
    // Don't stop at the end of the cut.
    xStop = cut.start().x() + diffX * cut.endSig();
    yStop = cut.start().y() + diffY * cut.endSig();
  }else
  {
    xStop = cut.stop().x();
    yStop = cut.stop().y();
  }
  
  // Compute the steps stepX and stepY along x and y.
  const std::size_t nSamples = cut.imgSignal().size();
  // We will work on the image while assuming an affine transformation.
  // Therefore, steps are computed in the pixel plane, regularly spaced from
  // the outer ellipse's center (cut.start()) to an outer point (cut.stop()).
  const float stepX = ( xStop - xStart ) / ( nSamples - 1.f );
  const float stepY = ( yStop - yStart ) / ( nSamples - 1.f );

  float x =  xStart;
  float y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    if ( x >= 1.f && x < src.cols-1 &&
         y >= 1.f && y < src.rows-1 )
    {
      //CCTAG_COUT_VAR2(x,y);
      // put pixel value to rectified signal
      cut.imgSignal()[i] = float(getPixelBilinear( src, x, y));
    }
    else
    {
      // push black
      cut.setOutOfBounds(true);
      break;
    }
    // Modify x and y to the next element.
    //CCTAG_COUT_VAR2(x,y);
    //CCTAG_COUT_VAR2(stepX,stepY);
    x += stepX;
    y += stepY;
    //CCTAG_COUT_VAR2(x,y);
  }
}

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
        std::size_t nSamplesInCut,
        float beginSig )
{
  // Collect all the 1D image signals from center to the outer points.
  cuts.reserve( outerPoints.size() );
  for( const cctag::DirectedPoint2d<Eigen::Vector3f> & outerPoint : outerPoints )
  {
    // Here only beginSig is set based on the input argument beginSig while endSig is set to 1.f as 
    // any type of cctags encodes, by construction, a 1D bar-code until the outer ellipse (image 
    // of the unit circle).
    cuts.emplace_back(center, outerPoint, beginSig, 1.f, nSamplesInCut );
    cctag::ImageCut & cut = cuts.back();
    cutInterpolated( cut, src);
    // Remove the cut from the vector if out of the image bounds.
    if ( cut.outOfBounds() )
    {
      CCTAG_COUT_VAR_OPTIM(cut.outOfBounds());
      cuts.pop_back();
    }
  }
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
float costSelectCutFun(
        const std::vector<float> & varCuts,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & outerPoints,
        const std::vector<std::size_t> & randomIdx,
        const float alpha)
{
  using namespace cctag::numerical;
  
  Eigen::Vector2f sumDeriv = Eigen::Vector2f::Zero();
  float sumVar = 0;
  for( const std::size_t i : randomIdx )
  {
    BOOST_ASSERT( i < varCuts.size() );
    
    //CCTAG_COUT_VAR(outerPoints[i].gradient());
    //CCTAG_COUT_VAR(sqrt(outerPoints[i].dX()*outerPoints[i].dX()
    //                  + outerPoints[i].dY()*outerPoints[i].dY()));
    
    sumDeriv += outerPoints[i].gradient(); // must be normalised. This normalisation is done during the CCTag construction.
    sumVar += varCuts[i];
  }

  const float ndir = sumDeriv.norm();

  return ndir - alpha * sumVar;
}

/**
 * @brief Select a subset of image cuts appropriate for the image center optimisation.
 * This selection aims at "maximizing" the variance of the image signal over all the 
 * selected cuts while ensuring a "good" distribution of the selected outer points
 * around the imaged center.
 *
 * @param[out] vSelectedCuts selected image cuts
 * @param[in] selectSize number of desired cuts to select
 * @param[in] collectedCuts all the collected cuts
 * @param[in] src source gray scale image (uchar)
 */
void selectCutCheapUniform( std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const float scale,
        const size_t numSamplesOuterEdgePointsRefinement)
{
  using namespace boost::numeric;
  using namespace boost::accumulators;
  using namespace cctag::numerical;

  selectSize = std::min( selectSize, collectedCuts.size() );

  std::vector<float> varCuts;
  varCuts.reserve(collectedCuts.size());
  for( const cctag::ImageCut & cut : collectedCuts )
  {
    accumulator_set< float, features< tag::variance > > acc;
    acc = std::for_each( cut.imgSignal().begin(), cut.imgSignal().end(), acc );
    varCuts.push_back( variance( acc ) );
  }
  
  const float varMax = *std::max_element(varCuts.begin(),varCuts.end());
  
  vSelectedCuts.clear();
  vSelectedCuts.reserve(selectSize);
  
  std::size_t sharpSize = 0;
  
  // Initialize vector of indices of sharp cuts
  std::vector<std::size_t> indToAdd;
  indToAdd.reserve(varCuts.size());
  
  for(std::size_t iCut = 0 ; iCut < varCuts.size() ; ++iCut)
  {
    ImageCut & cut = collectedCuts[iCut];
    cut.stop() = DirectedPoint2d<Eigen::Vector3f>(pointOnEllipse( outerEllipse, cut.stop() ), cut.stop().dX(), cut.stop().dY() );
    if ( outerEdgeRefinement(cut, src, scale, numSamplesOuterEdgePointsRefinement) )
      if ( varCuts[iCut]/varMax > 0.5f )
        indToAdd.push_back(iCut);
  }
  
  const float step = std::max(1.f, (float) indToAdd.size() / (float) ( selectSize ));
  
  for(std::size_t k=0 ; ; ++k)
  {
    if ( ( std::size_t(k*step) < indToAdd.size() ) && ( vSelectedCuts.size() < selectSize) )
    {
      ImageCut & cut = collectedCuts[indToAdd[std::size_t(k*step)]];
      vSelectedCuts.push_back( cut );
    }else{
      break;
    }
  }
}

/* Ugly -> perform an iterative optimization*/
bool outerEdgeRefinement(ImageCut & cut, const cv::Mat & src, float scale, std::size_t numSamplesOuterEdgePointsRefinement)
{
    // Subpixellic refinement of the outer edge points ///////////////////////////
    const float cutLengthOuterPointRefine = 3.f * sqrt(2.f) * scale; // with scale=2^i, i=0..nLevel
    const float halfWidth = cutLengthOuterPointRefine / 2.f;

    Eigen::Vector2f gradDirection = cut.stop().gradient()/cut.stop().gradient().norm();
    DirectedPoint2d<Eigen::Vector3f> cstop = cut.stop();
    
    Eigen::Vector2f hwgd = halfWidth * gradDirection;
    
    Point2d<Eigen::Vector3f> pStart( cstop(0)-hwgd(0), cstop(1)-hwgd(1) );
    
    const DirectedPoint2d<Eigen::Vector3f> pStop(
                                            cut.stop().x() + halfWidth*gradDirection(0),
                                            cut.stop().y() + halfWidth*gradDirection(1),
                                            cut.stop().dX(),
                                            cut.stop().dY());
    
    cctag::ImageCut cutOnOuterPoint(pStart, pStop, numSamplesOuterEdgePointsRefinement);
    cutInterpolated( cutOnOuterPoint, src);
    
    if (cutOnOuterPoint.outOfBounds())
      return false;
    
    
    std::vector<float> kernelA = { -0.0000f, -0.0003f, -0.1065f, -0.7863f, .0f, 0.7863f, 0.1065f, 0.0003f, 0.0000f }; // size = 9, sigma = 0.5
    std::vector<float> kernelB = { -0.0044f, -0.0540f, -0.2376f, -0.3450f, .0f, 0.3450f, 0.2376f, 0.0540f, 0.0044f }; // size = 9, sigma = 1
    std::vector<float> kernelC = { -0.0366f, -0.1113f, -0.1801f, -0.1594f, .0f, 0.1594f, 0.1801f, 0.1113f, 0.0366f }; // size = 9, sigma = 1.5

    std::vector<std::vector<float>> vKernels;
    vKernels.push_back(kernelA);
    vKernels.push_back(kernelB);
    vKernels.push_back(kernelC);
    
    std::map<float,float> res;
    
    for(size_t i=0; i<3 ; ++i)
    {
      res.insert(convImageCut(vKernels[i], cutOnOuterPoint));
    }
    // Get the location of the highest peak (last element)
    float maxLocation = res.rbegin()->second;
    
    float step = cutLengthOuterPointRefine/((float)numSamplesOuterEdgePointsRefinement-1.f);
    
    // Set the cut.stop() to its refined location.
    cut.stop() = DirectedPoint2d<Eigen::Vector3f>(
                    pStart.x() + step*maxLocation*gradDirection(0),
                    pStart.y() + step*maxLocation*gradDirection(1),
                    cut.stop().dX(),
                    cut.stop().dY());
    return true;
}

std::pair<float,float> convImageCut(const std::vector<float> & kernel, ImageCut & cut)
{
  //double guassOneD[] = { 0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044 };
  
  std::vector<float> output;
  std::size_t sizeCut = cut.imgSignal().size();
  std::size_t sizeKernel = kernel.size();
  output.resize(sizeCut);
  
  std::size_t halfSize = (sizeKernel-1)/2;
  
  //std::cout << "before = [" << std::endl;
  //for ( std::size_t i=0 ; i<sizeCut; ++i)
  //{
  //  std::cout << cut.imgSignal()[i] << " , " ;
  //}
  //std::cout << "]; " << std::endl;
  
  for ( std::ssize_t i=0 ; i<sizeCut; ++i)
  {
    float tmp = 0;
    for ( std::size_t j=0 ; j<sizeKernel; ++j)
    {
      if ( ssize_t(i-halfSize+j) < 0 )
        tmp += cut.imgSignal()[0]*kernel[j];
      else if( (i-halfSize+j) >=  sizeCut)
        tmp += cut.imgSignal()[sizeCut-1]*kernel[j];
      else
        tmp += cut.imgSignal()[i-halfSize+j]*kernel[j];
    }
    output[i] = tmp;
  }
  
  //std::cout << "convolved = [" << std::endl;
  //for ( std::size_t i=0 ; i<sizeCut; ++i)
  //{
  //  std::cout << output[i] << " , " ;
  //}
  //std::cout << "]; " << std::endl;
  
  //cut.imgSignal() = output;
  
  // Locate the maximum value.
  std::vector<float>::iterator maxValueIt = std::max_element(output.begin(), output.end());
  float itsLocation = (float) std::distance(output.begin(), maxValueIt);
  
  //CCTAG_COUT_VAR2(*maxValueIt, itsLocation);
  
  return std::pair<float,float>(*maxValueIt,itsLocation);// max value, its location
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
        const cv::Mat & src)
{
  Eigen::Matrix3f mInvHomography = mHomography.inverse();
  for( cctag::ImageCut & cut : vCuts )
  {
    extractSignalUsingHomography( cut, src, mHomography, mInvHomography);
  }
}

/**
 * @brief Compute an homography (up to a 2D rotation) based on its imaged origin [0,0,1]'
 * and its imaged unit circle (represented as an ellipse, assuming only quasi-affine transformation.
 *
 * @param[in] mEllipse ellipse matrix, projection of the unit circle
 * @param[in] center imaged center, projection of the origin
 * @param[out] mHomography computed homography
 */

void computeHomographyFromEllipseAndImagedCenter(
        const cctag::numerical::geometry::Ellipse & ellipse,
        const cctag::Point2d<Eigen::Vector3f> & center,
        Eigen::Matrix3f & mHomography)
 {
    using namespace cctag::numerical;

    Eigen::Matrix3f mCanonic(3,3);
    Eigen::Matrix3f mTCan(3,3);
    Eigen::Matrix3f mTInvCan(3,3);
    
    ellipse.getCanonicForm(mCanonic, mTCan, mTInvCan);
    
    // Get the center coordinates in the new canonical representation.
    float xc, yc;
    applyHomography(xc,yc, mTCan,center.x(), center.y());
            
    // Closed-form solution for the homography plan->image computation
    // The ellipse is supposed to be in its canonical representation

    // Matlab closed-form
    //H =
    //[ [     Q33,        Q22*xc*yc, -Q33*xc];
    //  [       0,      - Q11*xc^2 - Q33, -Q33*yc];
    //  [ -Q11*xc,        Q22*yc,    -Q33] ] ...
    // * diag([ ((Q22*Q33/Q11*(Q11*xc^2 + Q22*yc^2 + Q33)))^(1/2);Q33;(-Q22*(Q11*xc^2 + Q33))^(1/2)]);
    

    float Q11 = mCanonic(0,0);
    float Q22 = mCanonic(1,1);
    float Q33 = mCanonic(2,2);

    mHomography(0,0) = Q33;
    mHomography(1,0) = 0.f;
    mHomography(2,0) = -Q11*xc;

    mHomography(0,1) = Q22*xc*yc;
    mHomography(1,1) = -Q11*xc*xc-Q33;
    mHomography(2,1) = Q22*yc;

    mHomography(0,2) = -Q33*xc;
    mHomography(1,2) = -Q33*yc;
    mHomography(2,2) = -Q33;

    Eigen::Matrix3f mDiag;
    mDiag.setIdentity();
    mDiag(0,0) = sqrt((Q22*Q33/Q11*(Q11*xc*xc + Q22*yc*yc + Q33)));
    mDiag(1,1) = Q33;
    mDiag(2,2) = sqrt(-Q22*(Q11*xc*xc + Q33));

    for(int i=0; i < 3 ; ++i)
    {
      for(int j=0; j < 3 ; ++j)
      {
          mHomography(i,j) *= mDiag(j,j);
      }
    }

    mHomography = mTInvCan*mHomography;
}

/**
 * @brief Compute the optimal homography/imaged center based on the 
 * signal in the image and  the outer ellipse, supposed to be image the unit circle.
 * 
 * @param[in] tagIndex a sequence number for this tag
 * @param[out] mHomography optimal image->cctag homography
 * @param[out] optimalPoint optimal imaged center
 * @param[out] vCuts cuts holding the rectified 1D signals at the end of the optimization
 * @param[in] src source image
 * @param[in] ellipse outer ellipse (todo: is that already in the cctag object?)
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
        float & residual)
{
    using namespace cctag::numerical;

    // Visual debug
    CCTagVisualDebug::instance().newSession( "refineConicPts" );
    for(const cctag::ImageCut & cut : vCuts)
    {
        CCTagVisualDebug::instance().drawPoint( cut.stop(), cctag::color_red );
    }
    CCTagVisualDebug::instance().newSession( "centerOpt" );
    CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_green );

#ifdef WITH_CUDA
    if( cudaPipe ) {
        bool success = cudaPipe->imageCenterRetrieve(
            tagIndex,      // in
            optimalPoint,  // out
            residual,      // out
            mHomography,   // out
            params,
            cctag_pointer_buffer );

        if( not success ) {
            return false;
        }
    } else { // not CUDA
#endif // WITH_CUDA

        boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

        // A. Perform the optimization ///////////////////////////////////////////////

        // The neighbourhood size is 0.20*max(ellipse.a(),ellipse.b()), i.e. the max ellipse semi-axis
        float neighbourSize = params._imagedCenterNeighbourSize;

        std::size_t gridNSample = params._imagedCenterNGridSample; // todo: check must be odd 

        // The neighbourhood size is iteratively decreased, assuming the convexity of the 
        // cost function within it.
  
        float maxSemiAxis = std::max(outerEllipse.a(),outerEllipse.b());
  

  // Tests against synthetic experiments have shown that we do not reach a precision
  // better than 0.02 pixel.
  while ( neighbourSize*maxSemiAxis > 0.02 )       
  {
    if ( imageCenterOptimizationGlob( mHomography,   // out
                                      vCuts,         // out
                                      optimalPoint,  // out
                                      residual,      // out
                                      neighbourSize,
                                      src,
                                      outerEllipse,
                                      params ) )
    {
      CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_blue );
      neighbourSize /= float((gridNSample-1)/2) ;
    }else{
      return false;
    }
  }
  
  // Measure the time spent in the optimization
  boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
  boost::posix_time::time_duration d = tend - tstart;
  const float spendTime = d.total_milliseconds();
  DO_TALK( CCTAG_COUT_DEBUG( "Optimization result: " << optimalPoint << ", duration: " << spendTime ); )

#ifdef WITH_CUDA
    } // not CUDA
#endif // WITH_CUDA
    CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_red );
  
    // B. Get the signal associated to the optimal homography/imaged center //////
    {
        boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
        getSignals(vCuts,mHomography,src);
        boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
        boost::posix_time::time_duration d = tend - tstart;
    }
    
    // Residual normalization
    std::vector<std::size_t> correctCutIndices;
    correctCutIndices.reserve(vCuts.size());

    for(std::size_t iCut=0 ; iCut < vCuts.size() ; ++iCut) {
    if ( !vCuts[iCut].outOfBounds() )
        correctCutIndices.push_back(iCut);
    }

    // In barCode will be written the most frequent signal for every samples along
    // the cut.
    std::vector<float> barCode;
    const std::size_t signalSize = vCuts[0].imgSignal().size();
    barCode.resize(signalSize);

    std::vector<float> signalAlongX;
    signalAlongX.resize(correctCutIndices.size());

    for(std::size_t iSignal = 0 ; iSignal < signalSize ; ++iSignal)
    {
      for(std::size_t k=0 ; k <  correctCutIndices.size() ; ++k){
        signalAlongX[k] = vCuts[correctCutIndices[k]].imgSignal()[iSignal];
      }
      barCode[iSignal] = computeMedian( signalAlongX );
    }
    
    const float vMin = *std::min_element(barCode.begin(), barCode.end());
    const float vMax = *std::max_element(barCode.begin(), barCode.end());
    const float magnitude = vMax - vMin;

//    // Retains only the most reliable image cuts
//    std::map<float, std::size_t> mostReliableCuts;
//    for(std::size_t k=0 ; k <  correctCutIndices.size() ; ++k){
//      float squareDist = 0.f;
//      for(std::size_t iSignal = 0 ; iSignal < signalSize ; ++iSignal)
//        squareDist += std::pow(vCuts[correctCutIndices[k]].imgSignal()[iSignal]-barCode[iSignal],2);
//      mostReliableCuts.emplace( squareDist, correctCutIndices[k] );
//    }
//    
//    std::vector<cctag::ImageCut> outputs;
//    outputs.reserve(vCuts.size());
//    const std::size_t stop = correctCutIndices.size()/2 + 1;
//    std::size_t k=0;
//    for(const auto & iCut : mostReliableCuts)
//    {
//      outputs.push_back(vCuts[iCut.second]);
//      ++k;
//      if ( k > stop)
//        break;
//    }
//    vCuts = outputs;
    
    // Final normalized residual
    
    residual = sqrt(residual)/magnitude;
    if ( residual > 2.7f )
      return false;
    else
      return true;
}

/**
 * @brief Convex optimization of the imaged center within a point's neighbourhood.
 * 
 * @param[out] mHomography optimal homography from the pixel plane to the cctag plane.
 * @param[out] vCuts vector of the image cuts whose the signal has been rectified w.r.t. the computed mHomography
 * @param[in-out] center optimal imaged center
 * @param[out] minRes residual after optimization
 * @param[in] neighbourSize size of the neighbourhood to consider relatively to the outer ellipse dimensions
 * @param[in] gridNSample number of sample points along one dimension of the neighbourhood (e.g. grid)
 * @param[in] src source gray (uchar) image
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
        const cctag::numerical::geometry::Ellipse& outerEllipse,
        const cctag::Parameters & params )
{
    cctag::Point2d<Eigen::Vector3f> optimalPoint;
    Eigen::Matrix3f optimalHomography;
    bool hasASolution = false;

    using namespace cctag::numerical;

    const size_t gridNSample   = params._imagedCenterNGridSample;
  
    std::vector<cctag::Point2d<Eigen::Vector3f> > nearbyPoints;
    // A. Get all the grid point nearby the center /////////////////////////////
    getNearbyPoints( outerEllipse,  // in (ellipse)
                     center,        // in (Point2d)
                     nearbyPoints,  // out (vector<Point2d>)
                     neighbourSize, // in (float)
                     gridNSample,   // in (size_t)
                     GRID );        // in (enum)

    minRes = std::numeric_limits<float>::max();
    Eigen::Matrix3f mTempHomography;
 
    // For all points nearby the center ////////////////////////////////////////
    for(const cctag::Point2d<Eigen::Vector3f> & point : nearbyPoints)
    {
        CCTagVisualDebug::instance().drawPoint( point , cctag::color_green );

        // B. Compute the homography so that the back projection of 'point' is the
        // center, i.e. [0;0;1], and the back projected ellipse is the unit circle

        bool   readable = true;
        float res;

        {
            try
            {
                computeHomographyFromEllipseAndImagedCenter(
                    outerEllipse,     // in (ellipse)
                    point,            // in (Point2d)
                    mTempHomography); // out (matrix3x3)
            } catch(...) {
                continue; 
            }

            // C. Compute the 1D rectified signals of vCuts image cut based on the 
            // transformation mTempHomography.
            res = costFunctionGlob(mTempHomography, vCuts, src, readable );
        }
      
        // If at least one image cut has been properly read
        if ( readable )
        {       
        
                // Update the residual and the optimized parameters
                hasASolution = true;
                if ( res < minRes )
                {
                    minRes = res;
                    optimalPoint = point;
                    optimalHomography = mTempHomography;
                }
        } else { // not readable
                CCTAG_COUT_VAR_OPTIM(readable);
        }
    } // for(point : nearbyPoints)

    center = optimalPoint;
    mHomography = optimalHomography;
    
    return hasASolution;
}
  
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
        const std::size_t gridNSample,
        const NeighborType neighborType)
{
  nearbyPoints.clear();

  Eigen::Matrix3f mT = cctag::numerical::optimization::conditionerFromEllipse( ellipse );
  Eigen::Matrix3f mInvT = mT.inverse();
  
  cctag::numerical::geometry::Ellipse transformedEllipse(ellipse);
  cctag::viewGeometry::projectiveTransform( mInvT, transformedEllipse );
  neighbourSize *= std::max(transformedEllipse.a(),transformedEllipse.b());

  cctag::Point2d<Eigen::Vector3f> condCenter = center;
  cctag::numerical::optimization::condition(condCenter, mT);

  if ( neighborType == GRID )
  {
    const float gridWidth = neighbourSize;
    const float halfWidth = gridWidth/2.f;
    const float stepSize = gridWidth/(gridNSample-1);
 
    nearbyPoints.reserve(gridNSample*gridNSample);

    for(int i=0 ; i < gridNSample ; ++i)
    {
      for(int j=0 ; j < gridNSample ; ++j)
      {
        cctag::Point2d<Eigen::Vector3f> point(condCenter.x() - halfWidth + i*stepSize, condCenter.y() - halfWidth + j*stepSize );
        nearbyPoints.push_back(point);
      }
    }
  }
  cctag::numerical::optimization::condition(nearbyPoints, mInvT);
}

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
        bool & flag)
{
  flag = true;
  
  // Get the rectified signals along the image cuts
  getSignals( vCuts, mHomography, src);

  float res = 0;
  std::size_t resSize = 0;
  for( std::size_t i = 0; i < vCuts.size() - 1; ++i )
  {
    for( std::size_t j = i+1; j < vCuts.size(); ++j )
    {
      if ( !vCuts[i].outOfBounds() && !vCuts[j].outOfBounds() )
      {
        const auto& is = vCuts[i].imgSignal(), js = vCuts[j].imgSignal();
        assert(is.size() == js.size());
        for (size_t ii = 0; ii < js.size(); ++ii)
          res += std::pow(is[ii] - js[ii], 2);
        ++resSize;
      }
    }
  }
  // If no cut-pair has been found within the image bounds.
  if ( resSize == 0)
  {
    flag = false;
    return std::numeric_limits<float>::max();
  }else{
    // normalize, dividing by the total number of pairs in the image bounds.
    return res / resSize;
  }
}

/**
 * @brief Identify a marker:
 *   i) its imaged center is optimized: A. 1D image cuts are selected ; B. the optimization is performed 
 *   ii) the outer ellipse + the obtained imaged center delivers the image->cctag homography
 *   iii) the rectified 1D signals are read and deliver the ID via a nearest neighbour
 *        approach where the distance to the cctag bank's profiles used is the one described in [Orazio et al. 2011]
 * @param[in] tagIndex a sequence number assigned to this tag
 * @param[in] cctag whose center is to be optimized in conjunction with its associated homography.
 * @param[in] src original gray scale image (original scale, uchar)
 * @param[in] params set of parameters
 * @return status of the markers (c.f. all the possible status are located in CCTag.hpp) 
 */
int identify_step_1(
  int tagIndex,
  const CCTag & cctag,
  std::vector<cctag::ImageCut>& vSelectedCuts,
  const cv::Mat &  src,
  const cctag::Parameters & params)
{
  // Get the outer ellipse in its original scale, i.e. in src.
  const cctag::numerical::geometry::Ellipse & ellipse = cctag.rescaledOuterEllipse();
  // Get the outer points in their original scale, i.e. in src.
  const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & outerEllipsePoints = cctag.rescaledOuterEllipsePoints();

  // A. Pick a subsample of outer points ///////////////////////////////////////
  // Cheap (CPU only)
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
#endif
  
  // Sort outer points and then take a subsample
  std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > outerPoints;
  getSortedOuterPoints(ellipse, outerEllipsePoints, outerPoints, params._nSamplesOuterEllipse);
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d = t1 - t0;
  float spendTime;
  DO_TALK(

    spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in subsampling: " << spendTime << " ms");
  )
#endif
  

  assert ( outerPoints.size() >= 5 );
 
  // todo: next line deprec, associated to SUBPIX_EDGE_OPTIM, do not remove.
  const float cutLengthOuterPointRefine = std::min( ellipse.a(), ellipse.b() ) * 0.12f;

  // Visual debug
  for(const cctag::DirectedPoint2d<Eigen::Vector3f> & point : outerPoints)
  {
    CCTagVisualDebug::instance().drawPoint( Point2d<Eigen::Vector3f>(point.x(), point.y()), cctag::color_green );
  }

  // Set from where the rectified 1D signal should be read.
  // In fact, the white area located inside the inner ellipse does not hold
  // any information neither for the optimization nor for the reading.
  float startSig = 0.f;
  if (params._nCrowns == 3)
  {
    // Signal begin at 25% of the unit radius (for 3 black rings markers).
    // startOffset
    startSig = 1 - (2*params._nCrowns-1)*0.15f;
  }
  else if (params._nCrowns == 4)
  {
    startSig = 0.26f; // todo@Lilian
  }
  else
  {
    CCTAG_COUT("Error : unknown number of crowns");
  }
  // The "signal of interest" is located between startSig and 1.f (endSig in ImageCut)

#ifdef CCTAG_OPTIM
  t0 = boost::posix_time::microsec_clock::local_time();
#endif
  
  // B. Collect all cuts associated to all outer points ////////////////////////
  std::vector<cctag::ImageCut> cuts;
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    
    collectCuts( cuts, src, ellipse.center(), outerPoints, params._sampleCutLength, startSig);
    
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const float spendTime = d.total_milliseconds();
  }
  
#ifdef CCTAG_OPTIM
  t1 = boost::posix_time::microsec_clock::local_time();
  d = t1 - t0;
  DO_TALK(

    spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in collectCuts: " << spendTime << " ms");
  )
#endif

  if ( cuts.size() == 0 )
  {
    // Can happen when an object or the image frame is occluding a part of all available cuts.
    return status::no_collected_cuts;
  }
  
  // C. Select a sub sample of image cuts //////////////////////////////////////
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

    selectCutCheapUniform(
            vSelectedCuts,
            params._numCutsInIdentStep,
            ellipse,
            cuts,
            src,
            cctag.scale(),
            params._numSamplesOuterEdgePointsRefinement);
    
    DO_TALK( CCTAG_COUT_OPTIM("Initial cut selection"); )
    
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const float spendTime = d.total_milliseconds();
  }

  if ( vSelectedCuts.size() == 0 )
  {
    CCTAG_COUT_DEBUG("Unable to select any cut.");
    return status::no_selected_cuts; // todo: is class attributes the best option?
  }

  /* This is a fake return value. The important thing is to
   * distinguish this from the return value
   * status::no_collected_cuts
   */
  return status::id_reliable;
}
  
/**
 * @brief Identify a marker:
 *   i) its imaged center is optimized: A. 1D image cuts are selected ; B. the optimization is performed 
 *   ii) the outer ellipse + the obtained imaged center delivers the image->cctag homography
 *   iii) the rectified 1D signals are read and deliver the ID via a nearest neighbour
 *        approach where the distance to the cctag bank's profiles used is the one described in [Orazio et al. 2011]
 * @param[in] tagIndex a sequence number assigned to this tag
 * @param[inout] cctag whose center is to be optimized in conjunction with its associated homography.
 * @param[in] vSelectedCuts Cuts selected for this tag, list stays constant, signals are recomputed
 * @params[in] radiusRatios the Bank information
 * @param[in] src original gray scale image (original scale, uchar)
 * @param[inout] cudaPipe entry object for processing on the GPU
 * @param[in] params set of parameters
 * @return status of the markers (c.f. all the possible status are located in CCTag.hpp) 
 */
int identify_step_2(
  int tagIndex,
  CCTag & cctag,
  std::vector<cctag::ImageCut>& vSelectedCuts,
  const std::vector< std::vector<float> > & radiusRatios, // todo: directly use the CCTagBank
  const cv::Mat &  src,
  cctag::TagPipe* cudaPipe,
  const cctag::Parameters & params)
{
  // Get the outer ellipse in its original scale, i.e. in src.
  const cctag::numerical::geometry::Ellipse & ellipse = cctag.rescaledOuterEllipse();

  float residual = std::numeric_limits<float>::max();
    
  // C. Imaged center optimization /////////////////////////////////////////////
  // Expensive (GPU) Time bottleneck, the only function (including its sub functions) to be implemented on GPU
  // Note Inputs (CPU->GPU):
  //       i) src is already on GPU
  //       ii) _imgSignal in all ImageCut of vSelectedCuts do not need to be transfert,
  //       these signals will be collected inside the function.
  //       iii) cctag.homography(): 3x3 float homography, cctag.centerImg(): 2 floats (x,y), ellipse: (see Ellipse.hpp)
  // Begin GPU //////
  bool hasConverged = refineConicFamilyGlob(
                        tagIndex,
                        cctag.homography(),
                        cctag.centerImg(),
                        vSelectedCuts,
                        src,
                        cudaPipe,
                        ellipse,
                        params,
#ifdef WITH_CUDA
                        cctag.getNearbyPointBuffer(),
#else
                        nullptr,
#endif
                        residual
                        );
  
  cctag.setQuality(1.f/residual);
  
  // End GPU ////////
  // Note Outputs (GPU->CPU):
  //        The main amount of data to transfert is only that way and is 'vSelectedCuts', 
  //        the other outputs are of negligible size.
  //        All the ImageCut in vSelectedCuts including their attribute _imgSignal have to be transfer back to CPU.
  //        This operation is done once per marker. A maximum of 30 markers will be processed per frame. The 
  //        maximum number of cuts will be of 50. The maximum length of each _imgSignal will of 100*float.
  if( !hasConverged )
  {
    DO_TALK( CCTAG_COUT_DEBUG(ellipse); )
    CCTAG_COUT_VAR_DEBUG(cctag.centerImg());
    DO_TALK( CCTAG_COUT_DEBUG( "Optimization on imaged center failed to converge." ); )
    return status::opti_has_diverged;
  }
  
  MarkerID id = -1;

  std::size_t sizeIds = 6;
  IdSet idSet;
  idSet.reserve(sizeIds);

  bool identSuccessful = false;
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

    std::vector<std::list<float> > vScore;
    vScore.resize(radiusRatios.size());

  // D. Read the rectified 1D signals and retrieve the nearest ID(s) ///////////
  identSuccessful = orazioDistanceRobust( vScore, radiusRatios, vSelectedCuts, params._minIdentProba);
    
#ifdef VISUAL_DEBUG // todo: write a proper function in visual debug
  cv::Mat output;
  createRectifiedCutImage(vSelectedCuts, output);
  CCTagVisualDebug::instance().initBackgroundImage(output);
  CCTagVisualDebug::instance().newSession( "rectifiedSignal" + 
    std::to_string(CCTagVisualDebug::instance().getMarkerIndex()) );
  CCTagVisualDebug::instance().incrementMarkerIndex();
  // Back to session refineConicPts
  CCTagVisualDebug::instance().newSession( "refineConicPts" );
#endif // OPTIM_CENTER_VISUAL_DEBUG
    
#ifdef GRIFF_DEBUG
#error here
    // todo: clean and mode this block into a function.
    if( identSuccessful )
    {
#endif // GRIFF_DEBUG

      int maxSize = 0;
      int i = 0;
      int iMax = 0;

      for(const std::list<float> & lResult : vScore)
      {
        if (lResult.size() > maxSize)
        {
          iMax = i;
          maxSize = lResult.size();
        }
        ++i;
      }

      float score = 0;
#ifdef GRIFF_DEBUG
      assert( vScore.size() > 0 );
      assert( vScore.size() > iMax );
#endif // GRIFF_DEBUG
      for(const float & proba : vScore[iMax])
      {
        score += proba;
      }
      score /= vScore[iMax].size();

      // Set CCTag id
      cctag.setId( iMax );
      cctag.setIdSet( idSet );
      cctag.setRadiusRatios( radiusRatios[iMax] );

      // Push all the ellipses based on the obtained homography.
      try
      {

        Eigen::Matrix3f mInvH = cctag.homography().inverse();
        std::vector<cctag::numerical::geometry::Ellipse> & ellipses = cctag.ellipses();

        for(const float radiusRatio : cctag.radiusRatios())
        {
          cctag::numerical::geometry::Circle circle(1.f / radiusRatio);
          ellipses.emplace_back(mInvH.transpose()*circle.matrix()*mInvH);
        }

        // Push the outer ellipse
        ellipses.push_back(cctag.rescaledOuterEllipse());

        DO_TALK( CCTAG_COUT_VAR_DEBUG(cctag.id()); )
      }
      catch (...) // An exception can be thrown when a degenerate ellipse is computed.
      {
        return status::degenerate;
      }

      identSuccessful = (score > params._minIdentProba);
#ifdef GRIFF_DEBUG
    }
#endif // GRIFF_DEBUG

    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const float spendTime = d.total_milliseconds();
  }

  // Tell if the identification is reliable or not.
  if (identSuccessful)
  {
    return status::id_reliable;
  }
  else
  {
    return status::id_not_reliable;
  }
}

} // namespace identification
} // namespace cctag
