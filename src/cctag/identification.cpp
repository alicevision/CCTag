#include <cctag/identification.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/algebra/eig.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/viewGeometry/2DTransform.hpp>

#undef SUBPIX_EDGE_OPTIM
#include <cctag/SubPixEdgeOptimizer.hpp>

#ifdef USE_IMAGE_CENTER_OPT // undefined. Depreciated
#include <cctag/ImageCenterOptimizerCeres.hpp>
#include <cctag/ImageCenterOptimizer.hpp>
#include <cctag/LMImageCenterOptimizer.hpp>
#endif // USE_IMAGE_CENTER_OPT

#include <cctag/geometry/Cercle.hpp>
#include <cctag/talk.hpp>

#include <terry/sampler/all.hpp>

#include <openMVG/image/sample.hpp>

#include <opencv2/opencv.hpp>

#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/assert.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#undef NAIVE_SELECTCUT

#include <cmath>
#include <vector>

namespace cctag {
namespace identification {

bool orazioDistanceRobust(
        std::vector<std::list<double> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const double minIdentProba)
{
  BOOST_ASSERT( cuts.size() > 0 );

  using namespace cctag::numerical;
  using namespace boost::accumulators;

  typedef std::map<double, MarkerID> MapT;

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

  for( const cctag::ImageCut & cut : cuts )
  {
    if ( !cut.outOfBounds() )
    {
      MapT sortedId; // 6-nearest neighbours along with their affectation probability
      const std::size_t sizeIds = 6;
      IdSet idSet;
      idSet.reserve(sizeIds);

      // imgSig contains the rectified 1D signal.
      //boost::numeric::ublas::vector<double> imgSig( cuts.front().imgSignal().size() );
      const boost::numeric::ublas::vector<double> & imgSig = cut.imgSignal();

      // compute some statitics
      accumulator_set< double, features< /*tag::median,*/ tag::variance > > acc;
      // Put the image signal into the accumulator
      acc = std::for_each( imgSig.begin(), imgSig.end(), acc );

      const double medianSig = computeMedian( imgSig );
      // const double medianSig = computeMedian( boost::numeric::ublas::subrange(imgSig,startOffset, imgSig.size()) ); // todo : to remove

      const double varSig = boost::accumulators::variance( acc );

      accumulator_set< double, features< tag::mean > > accInf;
      accumulator_set< double, features< tag::mean > > accSup;
      for( std::size_t i = 0 ; i < imgSig.size(); ++i )
      {
        if( imgSig[i] < medianSig )
          accInf( imgSig[i] );
        else
          accSup( imgSig[i] );
      }
      const double muw = boost::accumulators::mean( accSup );
      const double mub = boost::accumulators::mean( accInf );

      // Find the nearest ID in rrBank
      const double stepX = (cut.endSig() - cut.beginSig()) / ( imgSig.size() - 1.0 );
      ///@todo vector<char>
      // vector of 1 or -1 values
      std::vector<double> digit( imgSig.size() );

  #ifdef GRIFF_DEBUG
      assert( rrBank.size() > 0 );
  #endif // GRIFF_DEBUG
      // Loop over imgSig values, compute and sum the difference between 
      // imgSig and digit (i.e. generated profile)
      for( std::size_t idc = 0; idc < rrBank.size(); ++idc )
      {
        // Compute profile - todo to be pre-computed
        double x = cut.beginSig();
        for( std::size_t i = 0; i < digit.size(); ++i )
        {
          std::ssize_t ldum = 0;
          for( std::size_t j = 0; j < rrBank[idc].size(); ++j )
          {
            if( 1.0 / rrBank[idc][j] <= x )
            {
              ++ldum;
            }
          }
          // set odd value to -1 and even value to 1
          digit[i] = - ( ldum % 2 ) * 2 + 1;
          
          x += stepX;
        }

        // compute distance to profile
        double distance = 0;
        for( std::size_t i = 0 ; i < imgSig.size() ; ++i )
        {
          distance += dis( imgSig[i], digit[i], mub, muw, varSig );
        }
        const double v = std::exp( -distance ); // todo: remove the exp and back transform the associated threshold
        sortedId[v] = idc;
      }

  #ifdef GRIFF_DEBUG
      assert( sortedId.size() > 0 );
  #endif // GRIFF_DEBUG
      int k = 0;
      BOOST_REVERSE_FOREACH( const MapT::const_iterator::value_type & v, sortedId )
      {
        if( k >= sizeIds ) break;
        std::pair< MarkerID, double > markerId;
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

      vScore[idSet.front().first].push_back(idSet.front().second);
    }
  }
  return true;
}

void centerScaleRotateHomography(
        cctag::numerical::BoundedMatrix3x3d & mHomography,
	const cctag::Point2dN<double> & center,
	const cctag::DirectedPoint2d<double> & point)
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;

  cctag::numerical::BoundedMatrix3x3d mInvHomography;
  invert( mHomography, mInvHomography );
  
  // Back projection of the image center
  Point2dN<double> backProjCenter = prec_prod< BoundedVector3d >( mInvHomography, center );
  {
    BoundedMatrix3x3d mTranslation;
    mTranslation( 0, 0 ) = 1.0;
    mTranslation( 0, 1 ) = 0.0;
    mTranslation( 0, 2 ) = backProjCenter.x();
    mTranslation( 1, 0 ) = 0.0;
    mTranslation( 1, 1 ) = 1.0;
    mTranslation( 1, 2 ) = backProjCenter.y();
    mTranslation( 2, 0 ) = 0.0;
    mTranslation( 2, 1 ) = 0.0;
    mTranslation( 2, 2 ) = 1.0;

    mHomography = prec_prod( mHomography, mTranslation );
    invert( mHomography, mInvHomography );
  }

  // New back projection
  backProjCenter = (Point2dN<double>) prec_prod< BoundedVector3d >( mInvHomography, cctag::Point2dN<double>(point.x(), point.y()) );
  const double scale = norm_2( subrange( backProjCenter, 0, 2 ) );
  BoundedVector3d rescaledBackProjCenter  = backProjCenter/scale;
  {
    BoundedMatrix3x3d mScaleRotation;
    mScaleRotation( 0, 0 ) = scale*rescaledBackProjCenter(0);
    mScaleRotation( 0, 1 ) = -scale*rescaledBackProjCenter(1);
    mScaleRotation( 0, 2 ) = 0.0;
    mScaleRotation( 1, 0 ) = scale*rescaledBackProjCenter(1);
    mScaleRotation( 1, 1 ) = scale*rescaledBackProjCenter(0);
    mScaleRotation( 1, 2 ) = 0.0;
    mScaleRotation( 2, 0 ) = 0.0;
    mScaleRotation( 2, 1 ) = 0.0;
    mScaleRotation( 2, 2 ) = 1.0;

    mHomography = prec_prod( mHomography, mScaleRotation );
  }
}


void createRectifiedCutImage(const std::vector<ImageCut> & vCuts, cv::Mat & output)
{
  output = cv::Mat(vCuts.size(), vCuts.front().imgSignal().size(), CV_8UC1);
  for(int i=0 ; i<vCuts.size() ; ++i)
  {
    const ImageCut & cut = vCuts[i];
    for(int j=0 ; j < cut.imgSignal().size() ; ++j)
    {
      output.at<uchar>(i,j) = (uchar) cut.imgSignal()(j);
    }
  }
}

/**
 * @brief Extract a rectified 1D signal along an image cut based on an homography.
 * 
 * @param[out] cut image cut holding the rectified image signal
 * @param[in] src source grayscale (uchar) image
 * @param[in] mHomography image->cctag homography
 * @param[in] mInvHomography cctag>image homography
 */
void extractSignalUsingHomography(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::numerical::BoundedMatrix3x3d & mHomography,
        const cctag::numerical::BoundedMatrix3x3d & mInvHomography)
{
  using namespace boost;
  using namespace boost::numeric::ublas;
  using namespace cctag::numerical;
  
  double xStart, xStop, yStart, yStop;
  
  double backProjStopX, backProjStopY;
  applyHomography(backProjStopX, backProjStopY, mInvHomography, cut.stop().x(), cut.stop().y());
  
  //CCTAG_COUT_VAR(Point2dN<double>(backProjStopX, backProjStopY).norm());
  
  // Check whether the signal to be collected start at 0.0 and stop at 1.0
  if ( cut.beginSig() != 0.0)
  {
    xStart = backProjStopX * cut.beginSig();
    yStart = backProjStopY * cut.beginSig();
  }else
  {
    xStart = 0;
    yStart = 0;
  }
  if ( cut.endSig() != 1.0)
  {
    xStop = backProjStopX * cut.endSig();
    yStop = backProjStopY * cut.endSig();
  }else
  {
    xStop  = backProjStopX;
    yStop  = backProjStopY; // xStop and yStop must not be normalised but the 
                            // norm([xStop;yStop]) is supposed to be close to 1.
  }

  // Compute the steps stepX and stepY along x and y.
  const std::size_t nSamples = cut.imgSignal().size();
  const double stepX = ( xStop - xStart ) / ( nSamples - 1.0 );
  const double stepY = ( yStop - yStart ) / ( nSamples - 1.0 );

  double xRes, yRes;

  double x =  xStart;
  double y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    
    // [xRes;yRes;1] ~= mHomography*[x;y;1.0]
    applyHomography(xRes, yRes, mHomography, x, y);

    if ( xRes >= 1.0 && xRes <= src.cols-1 &&
         yRes >= 1.0 && yRes <= src.rows-1 )
    {
      // Bilinear interpolation
      cut.imgSignal()(i) = getPixelBilinear( src, xRes, yRes);
    }
    else
    {
      cut.setOutOfBounds(true);
      // todo : add a break ?
    }
    
    x += stepX;
    y += stepY;
  }
}

/* depreciated */
void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedCut,
        const cv::Mat & src,
        cctag::numerical::BoundedMatrix3x3d & mHomography,
        std::size_t nSamples,
        const double begin,
        const double end)
{
  using namespace boost;
  using namespace boost::numeric::ublas;
  using namespace cctag::numerical;

  BOOST_ASSERT( rectifiedCut.imgSignal().size() == 0 );
  BOOST_ASSERT( end >= begin );
  
  // Check wheter the image signal size has been properly allocated.
  BOOST_ASSERT( nSamples == rectifiedCut.imgSignal().size() );

  nSamples = rectifiedCut.imgSignal().size();
  
  const double stepXi = ( end - begin ) / ( nSamples - 1.0 );

  rectifiedCut.start() = getHPoint( begin, 0.0, mHomography );
  rectifiedCut.stop() = cctag::DirectedPoint2d<double>( getHPoint( end, 0.0, mHomography ), 0.0, 0.0); // todo: here, the gradient information won't be required anymore.

  std::vector<std::size_t> idxNotInBounds;
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    const double xi = i * stepXi + begin;
    const cctag::Point2dN<double> hp = getHPoint( xi, 0.0, mHomography );

    if ( hp.x() >= 1.0 && hp.x() <= src.cols-1 &&
         hp.y() >= 1.0 && hp.y() <= src.rows-1 )
    {
      // openMVG::image::Sampler2d<openMVG::image::SamplerCubic> sampleFunctor;
      // //SamplerCubic
      // //SamplerSpline64
      // //SamplerCubic
      // //SamplerLinear
      // double pixVal = sampleFunctor.operator()<double>( src , float(hp.y()), float(hp.x()));
      
      // Bilinear interpolation
      rectifiedCut.imgSignal()(i) = getPixelBilinear( src, hp.x(), hp.y());
    }
    else
    {
      rectifiedCut.setOutOfBounds(true);
    }
  }
}

/// Code usable to apply a 1D gaussian filtering at the end of the extractSignalUsingHomography
///
//  double guassOneD[] = { 0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044 };
//  for( std::size_t i = 0; i < nSamples; ++i )
//  {
//    double tmp = 0;
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
 *             collected from cut.start() to cut.stop()
 * @param[in] src source gray scale image (uchar)
 */
void cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src)
{
  double xStart, yStart, xStop, yStop;
  const double diffX = cut.stop().x() - cut.start().x();
  const double diffY = cut.stop().y() - cut.start().y();
  
  // Check whether the signal to be collected start at 0.0 and stop at 1.0
  if ( cut.beginSig() != 0.0)
  {
    // Don't start at the beginning of the cut.
    xStart = cut.start().x() + diffX * cut.beginSig();
    yStart = cut.start().y() + diffY * cut.beginSig();
  }else
  {
    xStart = cut.start().x();
    yStart = cut.start().y();
  }
  if ( cut.endSig() != 1.0)
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
  const double stepX = ( xStop - xStart ) / ( nSamples - 1.0 );
  const double stepY = ( yStop - yStart ) / ( nSamples - 1.0 );

  double x =  xStart;
  double y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    if ( x >= 1.0 && x < src.cols-1 &&
         y >= 1.0 && y < src.rows-1 )
    {
      //CCTAG_COUT_VAR2(x,y);
      // put pixel value to rectified signal
      cut.imgSignal()(i) = double(getPixelBilinear( src, x, y));
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
 * Collect signals (image cuts) from center to outer ellipse points
 * 
 * @param[out] cuts collected cuts
 * @param[in] src source image (gray)
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
        const std::size_t nSamplesInCut,
        const double beginSig )
{
  // Collect all the 1D image signals from center to the outer points.
  cuts.reserve( outerPoints.size() );
  for( const cctag::DirectedPoint2d<double> & outerPoint : outerPoints )
  {
    // Here only beginSig is set based on the input argument beginSig while endSig is set to 1.0 as 
    // any type of cctags encodes, by construction, a 1D bar-code until the outer ellipse (image 
    // of the unit circle).
    cuts.push_back( cctag::ImageCut(center, outerPoint, beginSig, 1.0, nSamplesInCut) );
    cctag::ImageCut & cut = cuts.back();
    cutInterpolated( cut, src);
    // Remove the cut from the vector if out of the image bounds.
    if ( cut.outOfBounds() )
    {
      CCTAG_COUT_VAR(cut.outOfBounds());
      cuts.pop_back();
    }
  }
}

double costSelectCutFun(
        const std::vector<double> & varCuts,
        const std::vector< cctag::DirectedPoint2d<double> > & outerPoints,
        const boost::numeric::ublas::vector<std::size_t> & randomIdx,
        const double alpha)
{
  using namespace cctag::numerical;
  namespace ublas = boost::numeric::ublas;
  
  BoundedVector2d sumDeriv;
  double sumVar = 0;
  sumDeriv.clear();
  for( const std::size_t i : randomIdx )
  {
    BOOST_ASSERT( i < varCuts.size() );
    
    sumDeriv += outerPoints[i].gradient(); // must be normalised. This normalisation is done during the CCTag construction.
    sumVar += varCuts[i];
  }

  const double ndir = ublas::norm_2( sumDeriv );

  return ndir - alpha * sumVar;
}


void selectCut( std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const double cutLengthOuterPointRefine,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials )
{
  using namespace boost::numeric;
  using namespace boost::accumulators;

  selectSize = std::min( selectSize, collectedCuts.size() );

  std::vector<double> varCuts;
  varCuts.reserve( collectedCuts.size() );
  std::vector< cctag::DirectedPoint2d<double> > outerPoints;
  outerPoints.reserve( collectedCuts.size() );
  BOOST_FOREACH( const cctag::ImageCut & cut, collectedCuts )
  {
    accumulator_set< double, features< tag::variance > > acc;
    acc = std::for_each( cut.imgSignal().begin(), cut.imgSignal().end(), acc );

    varCuts.push_back( variance( acc ) );
    
    // Collect the normalized gradient over all outer points
    cctag::DirectedPoint2d<double> outerPoint( cut.stop() );
    double normGrad = sqrt(outerPoint.dX()*outerPoint.dX()* + outerPoint.dY()*outerPoint.dY());
    double dX = outerPoint.dX()/normGrad;
    double dY = outerPoint.dY()/normGrad;
    outerPoint.setDX(dX);
    outerPoint.setDY(dY);
    outerPoints.push_back( outerPoint );
  }

  // Maximize the sum of the variance while minimizing the norm of the sum of the normalized gradients 
  // over all collected cut stop.
  double Sm = std::numeric_limits<double>::max();
  ublas::vector<std::size_t> idxSelected;
  idxSelected.resize(selectSize);
  for( std::size_t i = 0; i < cutsSelectionTrials; ++i )
  {
    ublas::vector<std::size_t> randomIdx = boost::numeric::ublas::subrange( 
            cctag::numerical::randperm< ublas::vector<std::size_t> >( collectedCuts.size() ), 0, selectSize );
    
    double cost = costSelectCutFun( varCuts, outerPoints, randomIdx );
    if ( cost < Sm )
    {
      Sm = cost;
      idxSelected = randomIdx;
    }
  }
  
  // Ordered map to get variance from the higher value to the lower
  typedef std::multimap< double, cctag::ImageCut *, std::greater<double> > MapT;
  MapT mapVar;

  BOOST_FOREACH( const std::size_t i, idxSelected )
  {
    cctag::ImageCut & cut = collectedCuts[i];
    std::pair<double, cctag::ImageCut*> v( varCuts[i], &cut );
    mapVar.insert( v );
  }

#ifdef SUBPIX_EDGE_OPTIM // undefined. Depreciated
  // half size of the segment used to refine the external point of cut
  //const double halfWidth = cutLengthOuterPointRefine / 2.0;
#endif // SUBPIX_EDGE_OPTIM

  std::size_t i = 0;
  vSelectedCuts.reserve( selectSize );
  BOOST_FOREACH( MapT::value_type & v, mapVar )
  {
    cctag::ImageCut & cut = *v.second;
    BOOST_ASSERT( cut.stop().x() >= 0 && cut.stop().x() < src.cols );
    BOOST_ASSERT( cut.stop().y() >= 0 && cut.stop().y() < src.rows );
    BOOST_ASSERT( cut.stop().x() >= 0 && cut.stop().x() < src.cols );
    BOOST_ASSERT( cut.stop().y() >= 0 && cut.stop().y() < src.rows );

#ifdef SUBPIX_EDGE_OPTIM
    const double halfWidth = cutLengthOuterPointRefine / 2.0;
    cctag::numerical::BoundedVector2d gradDirection = cctag::numerical::unit( cut.stop().gradient() );
    BOOST_ASSERT( norm_2( gradDirection ) != 0 );

    const Point2dN<double> pStart( Point2dN<double>(cut.stop()) - halfWidth * gradDirection);
    const DirectedPoint2d<double> pStop(
                                          Point2dN<double>(
                                                  cut.stop().x() + halfWidth*gradDirection(0),
                                                  cut.stop().y() + halfWidth*gradDirection(1)),
                                          cut.stop().dX(),
                                          cut.stop().dY());
    
    cctag::ImageCut cutOnOuterPoint(pStart, pStop, numSamplesOuterEdgePointsRefinement);
    cutInterpolated( cutOnOuterPoint, src);
    
    if ( !cutOnOuterPoint.outOfBounds() )
    {
      SubPixEdgeOptimizer optimizer( cutOnOuterPoint );
      cctag::Point2dN<double> refinedPoint = 
        optimizer(
                halfWidth,
                cut.stop().x(),
                cutOnOuterPoint.imgSignal()[0],
                cutOnOuterPoint.imgSignal()[cutOnOuterPoint.imgSignal().size()-1] );

      // Take cuts the didn't diverge too much
      if ( cctag::numerical::distancePoints2D( cut.stop(), refinedPoint ) < halfWidth )
      {
        // x and y are refined. The gradient is kept as it was because the refinement.
        cut.stop().setX( refinedPoint.x() );
        cut.stop().setY( refinedPoint.y() );
        //cut.stop() = cctag::DirectedPoint2d<double>(refinedPoint.x(),refinedPoint.y(),cut.stop().dX(), cut.stop().dY());
        vSelectedCuts.push_back( cut );
      }
    }
#else // SUBPIX_EDGE_OPTIM
    vSelectedCuts.push_back( cut );
#endif // SUBPIX_EDGE_OPTIM

    ++i;
    if( vSelectedCuts.size() >= selectSize )
    {
      break;
    }
  }
}

void selectCutNaive( // depreciated: dx and dy are not accessible anymore -> use DirectedPoint instead
        std::vector< cctag::ImageCut > & vSelectedCuts,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize,
        const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy )
{
  using namespace boost::numeric;
  using namespace boost::accumulators;

  selectSize = std::min( selectSize, collectedCuts.size() );

  std::vector<double> vGrads;
  vGrads.reserve( collectedCuts.size() );
  for( int i=0 ; i < collectedCuts.size() ; ++i)
  {
    ublas::bounded_vector<double,2> gradient;
    gradient(0) = dx.at<short>( collectedCuts[i].stop().y(), collectedCuts[i].stop().x() );
    gradient(1) = dy.at<short>( collectedCuts[i].stop().y(), collectedCuts[i].stop().x() );
    vGrads.push_back(ublas::norm_2(gradient));
    //CCTAG_COUT_VAR(vGrads.back());
  }
  // Statistics on vGrads
  accumulator_set< double, features< tag::median > > acc;
  // Compute the median value
  acc = std::for_each( vGrads.begin(), vGrads.end(), acc );
  const double medianValue = boost::accumulators::median( acc );
  //CCTAG_COUT_VAR(medianValue);

  const std::size_t step = std::max( 1, int( collectedCuts.size()/2 -1 ) / int(selectSize) );
  
  // Select all cuts whose stop gradient is greater than the median value
  std::size_t iStep = 0;
          
  for( int i=0 ; i < collectedCuts.size() ; ++i)
  {
    ublas::bounded_vector<double,2> gradient;
    gradient(0) = dx.at<short>( collectedCuts[i].stop().y(), collectedCuts[i].stop().x() );
    gradient(1) = dy.at<short>( collectedCuts[i].stop().y(), collectedCuts[i].stop().x() );
    
    if ( ublas::norm_2(gradient) > medianValue )
    {
      if ( iStep == step )
      {
        //cctag::Point2dN<double> refinedPoint(collectedCuts[i].stop());
        prSelection.push_back( collectedCuts[i].stop() );
        vSelectedCuts.push_back( collectedCuts[i] );
        iStep = 0;
      }else
      {
        ++iStep;
      }
      
    }
    
    //if( vSelectedCuts.size() >= selectSize )
    //{
    //  break;
    //}
  }
}

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
        const cv::Mat & src)
{
  cctag::numerical::BoundedMatrix3x3d mInvHomography;
  cctag::numerical::invert(mHomography, mInvHomography);
  for( cctag::ImageCut & cut : vCuts )
  {
    extractSignalUsingHomography( cut, src, mHomography, mInvHomography);
  }
}

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
        cctag::numerical::BoundedMatrix3x3d & mHomography)
 {
    using namespace cctag::numerical;
    using namespace boost::numeric::ublas;
  
  cctag::numerical::BoundedMatrix3x3d mA;
  invert( mEllipse, mA );
  cctag::numerical::BoundedMatrix3x3d mO = outer_prod( center, center );
  diagonal_matrix<double> vpg;

  cctag::numerical::BoundedMatrix3x3d mVG;
  // Compute eig(inv(A),center*center')
  eig( mA, mO, mVG, vpg ); // Warning : compute GENERALIZED eigvalues, take 4 parameters !
                           // eig(a,b,c) compute eigenvalues of a, call a different 
                           // routine in lapack.

  cctag::numerical::Matrixd u, v;
  diagonal_matrix<double> s( 3, 3 );
  double vmin = std::abs( vpg( 0, 0 ) );
  std::size_t imin = 0;

  // Find minimum of the generalized eigen values
  for( std::size_t i = 1; i < vpg.size1(); ++i )
  {
    double v = std::abs( vpg( i, i ) );
    if ( v < vmin )
    {
      vmin = v;
      imin = i;
    }
  }

  svd( mA - vpg( imin, imin ) * mO, u, v, s );

  for( std::size_t i = 0; i < s.size1(); ++i )
  {
    BOOST_ASSERT( s( i, i ) >= 0.0 );
    s( i, i ) = std::sqrt( s( i, i ) );
  }

  cctag::numerical::BoundedMatrix3x3d mU = prec_prod( u, s );

  column( mHomography, 0 ) = column( mU, 0 );
  column( mHomography, 1 ) = column( mU, 1 );
  column( mHomography, 2 ) = cross( column( mU, 0 ), column( mU, 1 ) );
  
  // The circular points have been computed.
  // The following ensures that the back projection is at the origin (through a translation)
  // and the the back projected outer ellipse is of unit radius.
  
  // Translation part
  cctag::numerical::BoundedMatrix3x3d mInvHomography;
  invert( mHomography, mInvHomography );
  
  // Back projection of the image center
  Point2dN<double> backProjCenter = prec_prod< BoundedVector3d >( mInvHomography, center );
  BoundedMatrix3x3d mTranslation; // todo Initialize with eye(3).
  mTranslation( 0, 0 ) = 1.0;
  mTranslation( 0, 1 ) = 0.0;
  mTranslation( 0, 2 ) = backProjCenter.x();
  mTranslation( 1, 0 ) = 0.0;
  mTranslation( 1, 1 ) = 1.0;
  mTranslation( 1, 2 ) = backProjCenter.y();
  mTranslation( 2, 0 ) = 0.0;
  mTranslation( 2, 1 ) = 0.0;
  mTranslation( 2, 2 ) = 1.0;

  mHomography = prec_prod( mHomography, mTranslation );
  
  // Scaling part
  cctag::numerical::geometry::Ellipse backProjectedOuterEllipse(mEllipse);
  cctag::viewGeometry::projectiveTransform( mHomography, backProjectedOuterEllipse );
  // todo require to rebuild an Ellipse object, pass ellipse instead of matEllipse as argument.
  
  const double scale = ( backProjectedOuterEllipse.a() + backProjectedOuterEllipse.b() ) / 2.0;
  //CCTAG_COUT_VAR(backProjectedOuterEllipse);

  BoundedMatrix3x3d mScale;
  mScale( 0, 0 ) = scale; // todo Initialize with eye(3).
  mScale( 0, 1 ) = 0.0;
  mScale( 0, 2 ) = 0.0;
  mScale( 1, 0 ) = 0.0;
  mScale( 1, 1 ) = scale;
  mScale( 1, 2 ) = 0.0;
  mScale( 2, 0 ) = 0.0;
  mScale( 2, 1 ) = 0.0;
  mScale( 2, 2 ) = 1.0;

  mHomography = prec_prod( mHomography, mScale );
  backProjectedOuterEllipse = cctag::numerical::geometry::Ellipse(mEllipse);
  cctag::viewGeometry::projectiveTransform( mHomography, backProjectedOuterEllipse );
  //CCTAG_COUT_VAR2(backProjectedOuterEllipse.a(), backProjectedOuterEllipse.b());
  //CCTAG_COUT_VAR(backProjectedOuterEllipse);
  
  // Normalize
  mHomography = mHomography/mHomography(2,2);
}

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
 * @param[in] outerEllipse outer ellipse (todo: is that already in the cctag object?)
 * @return true if the optimization has converged, false otherwise.
 */
bool refineConicFamilyGlob(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & vCuts, 
        const std::size_t nSamples,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & outerEllipse)
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;

  cctag::numerical::BoundedMatrix3x3d & mHomography = cctag.homography();
  Point2dN<double> & optimalPoint = cctag.centerImg();

  BOOST_ASSERT( vOuterPoints.size() > 0 );

  // Debug
  CCTagVisualDebug::instance().newSession( "refineConicPts" );
  for(const cctag::ImageCut & cut : vCuts)
  {
    CCTagVisualDebug::instance().drawPoint( cctag::Point2dN<double>(cut.stop().x(), cut.stop().y()), cctag::color_red );
  }

  //center = outerEllipse.center(); // todo: might be better instead of the center of the rescaled one.

  CCTagVisualDebug::instance().newSession( "centerOpt" );
  CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_green );

  boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

  // Perform the optimization
  std::vector<cctag::Point2dN<double> > nearbyPoints;

  double neighbourSize = 0.15;
  double residual;

  std::size_t gridNSample = 4;

  // The neighbourhood size is iteratively decreased, assuming the convexity of the 
  // cost function within it.
  while ( neighbourSize > 1e-4 )
  {
    if ( imageCenterOptimizationGlob(mHomography,vCuts,optimalPoint,residual,neighbourSize,gridNSample,src,outerEllipse) )
    {
      CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_blue );
      neighbourSize /= gridNSample ;
    }else{
      return false;
    }
  }

  // Measure the time spent in the optimization
  boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
  boost::posix_time::time_duration d = tend - tstart;
  const double spendTime = d.total_milliseconds();
  DO_TALK( CCTAG_COUT_DEBUG( "Optimization result: " << optimalPoint << ", duration: " << spendTime ); )

  CCTagVisualDebug::instance().drawPoint( optimalPoint, cctag::color_red );

  CCTAG_COUT_VAR(optimalPoint);
  CCTAG_COUT_VAR(cctag.centerImg());
  
  // Get the signal associated to the optimal homography/imaged center.
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    getSignals(vCuts,mHomography,src);
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
  }
  return true;
}

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
        const cctag::numerical::geometry::Ellipse & outerEllipse)
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;
  
    std::vector<cctag::Point2dN<double> > nearbyPoints;
    getNearbyPoints(outerEllipse, center, nearbyPoints, neighbourSize, gridNSample, GRID);
    
    minRes = std::numeric_limits<double>::max();
    cctag::Point2dN<double> optimalPoint;
    BoundedMatrix3x3d optimalHomography;
    BoundedMatrix3x3d mTempHomography;
    
    bool hasASolution = false;

#ifdef OPTIM_CENTER_VISUAL_DEBUG
    int k = 0;
#endif // OPTIM_CENTER_VISUAL_DEBUG    
    for(const cctag::Point2dN<double> & point : nearbyPoints)
    {
      CCTagVisualDebug::instance().drawPoint( point , cctag::color_green );
      computeHomographyFromEllipseAndImagedCenter( outerEllipse.matrix(), point, mTempHomography);
      
      // Back projection of the image center
      BoundedMatrix3x3d mInvHomography;
      invert( mTempHomography, mInvHomography );
      Point2dN<double> backProjCenter = prec_prod< BoundedVector3d >( mInvHomography, center );
      //CCTAG_COUT_VAR(backProjCenter);
      
      bool readable = true;
      
      double res = costFunctionGlob(mTempHomography, vCuts, src, readable);
      
      if ( readable )
      {
#ifdef OPTIM_CENTER_VISUAL_DEBUG
        cv::Mat output;
        createRectifiedCutImage(vCuts, output);
        cv::imwrite("/home/lilian/data/temp/" + std::to_string(k) + ".png", output);
        ++k;
#endif // OPTIM_CENTER_VISUAL_DEBUG        
        
        hasASolution = true;
        if ( res < minRes )
        {
          minRes = res;
          optimalPoint = point;
          optimalHomography = mTempHomography;
        }
      }else
      {
        CCTAG_COUT_VAR(readable);
      }
    }
    center = optimalPoint;
    CCTAG_COUT_VAR(center);
    mHomography = optimalHomography;
    
    return hasASolution;
}
  
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
        const NeighborType neighborType)
{
  nearbyPoints.clear();

  cctag::numerical::BoundedMatrix3x3d mT = cctag::numerical::optimization::conditionerFromEllipse( ellipse );

  cctag::Point2dN<double> condCenter = center;
  cctag::numerical::optimization::condition(condCenter, mT);

  cctag::numerical::BoundedMatrix3x3d mInvT;
  cctag::numerical::invert_3x3(mT,mInvT);

  if ( neighborType == GRID )
  {
    const double gridWidth = neighbourSize;
    const double halfWidth = gridWidth/2.0;
    const double stepSize = gridWidth/(gridNSample-1);
    
    nearbyPoints.reserve(gridNSample*gridNSample);

    for(int i=0 ; i < gridNSample ; ++i)
    {
      for(int j=0 ; j < gridNSample ; ++j)
      {
        cctag::Point2dN<double> point(condCenter.x() - halfWidth + i*stepSize, condCenter.y() - halfWidth + j*stepSize );
        nearbyPoints.push_back(point);
      }
    }
  }
  cctag::numerical::optimization::condition(nearbyPoints, mInvT);
}

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
        bool & flag)
{
  flag = true;
  
  getSignals( vCuts, mHomography, src);

  double res = 0;
  std::size_t resSize = 0;
  for( std::size_t i = 0; i < vCuts.size() - 1; ++i )
  {
    for( std::size_t j = i+1; j < vCuts.size(); ++j )
    {
      if ( !vCuts[i].outOfBounds() && !vCuts[j].outOfBounds() )
      {
        //  += (signalCutI(0)-signalCutJ(0))*(signalCutI(0)-signalCutJ(0)) + ... + (signalCutI(end)-signalCutJ(end))*(signalCutI(end)-signalCutJ(end)) // GPU
        res += std::pow( norm_2( vCuts[i].imgSignal() - vCuts[j].imgSignal() ), 2 );
        ++resSize;
      }
    }
  }
  // If no cut-pair has been found within the image bounds.
  if ( resSize == 0)
  {
    flag = false;
    return std::numeric_limits<double>::max();
  }else{
    // normalize, dividing by the total number of pairs in the image bounds.
    return res /= resSize;
  }
}

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
  const std::vector< std::vector<double> > & radiusRatios, ///@todo directly use the bank
  const cv::Mat & src,
  const cctag::Parameters & params)
{
  // Get the outer ellipse in its orignal scale, i.e. in src.
  const cctag::numerical::geometry::Ellipse & ellipse = cctag.rescaledOuterEllipse();
  const std::vector< cctag::DirectedPoint2d<double> > & outerEllipsePoints = cctag.rescaledOuterEllipsePoints();
  // outerEllipsePoints can be changed in the edge point refinement - not const - todo@Lilian - save their modifications
  // in the CCTag instance just above _rescaledOuterEllipsePoints.

  // Take 100 edge points around outer ellipse.
  const std::size_t nOuterPoints = std::min( std::size_t(100), outerEllipsePoints.size() );
  std::size_t step = std::size_t( outerEllipsePoints.size() / ( nOuterPoints - 1 ) );
  std::vector< cctag::DirectedPoint2d<double> > outerPoints;

  outerPoints.reserve( nOuterPoints );
  for( std::size_t i = 0; i < outerEllipsePoints.size(); i += step )
  {
    outerPoints.push_back( outerEllipsePoints[i] );
  }

  ///@todo Check if a & b sorted (cf. ellipse2param)
  const double cutLengthOuterPointRefine = std::min( ellipse.a(), ellipse.b() ) * 0.12;

  for(const cctag::DirectedPoint2d<double> & point : outerPoints)
  {
    CCTagVisualDebug::instance().drawPoint( Point2dN<double>(point.x(), point.y()), cctag::color_green );
    //todo: templater la fonction draw au lieu de reconstruire
  }

  double startSig = 0.0;

  if (params._nCrowns == 3)
  {
    // Signal begin at 25% of the unit radius (for 3 black rings markers).
    // startOffset
    startSig = 1 - (2*params._nCrowns-1)*0.15;	// Considering 6 radius.
  }
  else if (params._nCrowns == 4)
  {
    startSig = 0.26;
  }
  else
  {
    CCTAG_COUT("Error : unknown number of crowns");
  }

#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t0(boost::posix_time::microsec_clock::local_time());
#endif
  
  std::vector<cctag::ImageCut> cuts;
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    
    collectCuts( cuts, src, ellipse.center(), outerPoints, params._sampleCutLength, startSig);
    
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    //CCTAG_TCOUT( "CollectCuts, duration: " << spendTime );
  }
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());
  boost::posix_time::time_duration d = t1 - t0;
  double spendTime;
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
  
  std::vector< cctag::ImageCut > vSelectedCuts;

  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

#ifdef NAIVE_SELECTCUT
    'depreciated: dx and dy are not accessible anymore -> use DirectedPoint instead'
    selectCutNaive( vSelectedCuts, prSelection, params._numCutsInIdentStep, cuts, src, 
          dx, dy ); 
    DO_TALK( CCTAG_COUT_OPTIM("Naive cut selection"); )
#else
    selectCut(
            vSelectedCuts,
            params._numCutsInIdentStep,
            cuts,
            src,
            cutLengthOuterPointRefine,
            params._numSamplesOuterEdgePointsRefinement,
            params._cutsSelectionTrials
            );
    DO_TALK( CCTAG_COUT_OPTIM("Initial cut selection"); )
#endif
    
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
  }
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());
  DO_TALK(

    d = t2 - t1;
    spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in selectCut: " << spendTime << " ms");
  )
#endif

  if ( vSelectedCuts.size() == 0 )
  {
    //
    CCTAG_COUT_DEBUG("Unable to select any cut.");
    return status::no_selected_cuts; // todo: is class attributes the best option?
  }

  std::vector< cctag::ImageCut > vCuts;
  
  {
//    bool hasConverged = refineConicFamily( cctag, vCuts, params._sampleCutLength, src, ellipse, prSelection, params._useLMDif );
//    if( !hasConverged )
//    {
//      DO_TALK( CCTAG_COUT_DEBUG(ellipse); )
//      CCTAG_COUT_VAR_DEBUG(cctag.centerImg());
//      DO_TALK( CCTAG_COUT_DEBUG( "Optimization on imaged center failed to converge." ); )
//      return status::opti_has_diverged;
//    }
    
    bool hasConverged = refineConicFamilyGlob( cctag, vSelectedCuts, params._sampleCutLength, src, ellipse);
    if( !hasConverged )
    {
      DO_TALK( CCTAG_COUT_DEBUG(ellipse); )
      CCTAG_COUT_VAR_DEBUG(cctag.centerImg());
      DO_TALK( CCTAG_COUT_DEBUG( "Optimization on imaged center failed to converge." ); )
      return status::opti_has_diverged;
    }
    
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t3(boost::posix_time::microsec_clock::local_time());
  DO_TALK(

    d = t3 - t2;
    spendTime = d.total_milliseconds();
    CCTAG_COUT_OPTIM("Time in refineConicFamily: " << spendTime << " ms");
  )
#endif
  }
  
  MarkerID id = -1;

  std::size_t sizeIds = 6;
  IdSet idSet;
  idSet.reserve(sizeIds);

  bool idFinal = false;
  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

    std::vector<std::list<double> > vScore;
    vScore.resize(radiusRatios.size());

#ifdef INITIAL_1D_READING // not used anymore: not defined
                          // v0.0 for the identification: use a single cut, average of the rectified cut, to read the id.
      idFinal = orazioDistance( idSet, radiusRatios, vSelectedCuts, startOffset, params._minIdentProba, sizeIds);
      // If success
      if ( idFinal )
      {
        // Set CCTag id
        id = idSet.front().first;
        cctag.setId( id );
        cctag.setIdSet( idSet );
        cctag.setRadiusRatios( radiusRatios[id] );
      }
      else
      {
        DO_TALK( CCTAG_COUT_DEBUG("Not enough quality in IDENTIFICATION"); )
      }
#else // INITIAL_1D_READING // used
      // v0.1 for the identification: use the most redundant id over all the rectified cut.
      idFinal = orazioDistanceRobust( vScore, radiusRatios, vSelectedCuts, params._minIdentProba);
#ifdef GRIFF_DEBUG
      if( idFinal )
      {
#endif // GRIFF_DEBUG

        int maxSize = 0;
        int i = 0;
        int iMax = 0;

        BOOST_FOREACH(const std::list<double> & lResult, vScore)
        {
          if (lResult.size() > maxSize)
          {
            iMax = i;
            maxSize = lResult.size();
          }
          ++i;
        }

        double score = 0;
#ifdef GRIFF_DEBUG
        assert( vScore.size() > 0 );
        assert( vScore.size() > iMax );
#endif // GRIFF_DEBUG
        BOOST_FOREACH(const double & proba, vScore[iMax])
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
          using namespace boost::numeric::ublas;
          
          bounded_matrix<double, 3, 3> mInvH;
          cctag::numerical::invert(cctag.homography(), mInvH);
          std::vector<cctag::numerical::geometry::Ellipse> & ellipses = cctag.ellipses();

          for(const double radiusRatio : cctag.radiusRatios())
          {
            cctag::numerical::geometry::Cercle circle(1.0 / radiusRatio);
            ellipses.push_back(cctag::numerical::geometry::Ellipse(
                    prec_prod(trans(mInvH), prec_prod<bounded_matrix<double, 3, 3> >(circle.matrix(), mInvH))));
          }

          // Push the outer ellipse
          ellipses.push_back(cctag.rescaledOuterEllipse());

          DO_TALK( CCTAG_COUT_VAR_DEBUG(cctag.id()); )
        }
        catch (...) // An exception can be thrown when a degenerate ellipse is computed.
        {
          return status::degenerate;
        }
        
        idFinal = (score > params._minIdentProba);
#ifdef GRIFF_DEBUG
      }
#endif // GRIFF_DEBUG
      
#endif // INITIAL_1D_READING

    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
  }

  // Tell if the identification is reliable or not.
  if (idFinal)
  {
    return status::id_reliable;
  }
  else
  {
    return status::id_not_reliable;
  }
}

/* depreciated */
bool orazioDistance( IdSet& idSet, const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds)
{
  BOOST_ASSERT( cuts.size() > 0 );

  using namespace cctag::numerical;
  using namespace boost::accumulators;

  typedef std::map<double, MarkerID> MapT;
  MapT sortedId;

  if ( cuts.size() == 0 )
  {
    return false;
  }
  // isig contains 1D signal on line.
  boost::numeric::ublas::vector<double> isig( cuts.front().imgSignal().size() );
  BOOST_ASSERT( isig.size() - startOffset > 0 );

  // Sum all cuts to isig
  for( std::size_t i = 0; i < isig.size(); ++i )
  {
    double& isigCurrent = isig(i);
    isigCurrent = 0.0;
    BOOST_FOREACH( const cctag::ImageCut & cut, cuts )
    {
      isigCurrent += cut.imgSignal()( i );
    }
  }

  //CCTAG_TCOUT_VAR(isig);

  // compute some statitics
  accumulator_set< double, features< /*tag::median,*/ tag::variance > > acc;
  // put sub signal into the statistical tool
  acc = std::for_each( isig.begin()+startOffset, isig.end(), acc );

  //CCTAG_TCOUT_VAR(boost::numeric::ublas::subrange(isig,startOffset, isig.size()));

  //const double mSig = boost::accumulators::median( acc );
  const double mSig = computeMedian( boost::numeric::ublas::subrange(isig,startOffset, isig.size()) );

  //CCTAG_TCOUT("Median of the signal : " << mSig);

  const double varSig = boost::accumulators::variance( acc );

  accumulator_set< double, features< tag::mean > > accInf;
  accumulator_set< double, features< tag::mean > > accSup;
  for( std::size_t i = startOffset; i < isig.size(); ++i )
  {
    if( isig[i] < mSig )
      accInf( isig[i] );
    else
      accSup( isig[i] );
  }
  const double muw = boost::accumulators::mean( accSup );
  const double mub = boost::accumulators::mean( accInf );

  //CCTAG_TCOUT(muw);
  //CCTAG_TCOUT(mub);

  // find the nearest ID in rrBank
  const double stepXi = 1.0 / ( isig.size() + 1.0 ); /// @todo lilian +1 ??
  ///@todo vector<char>
  // vector of 1 or -1 values
  std::vector<double> digit( isig.size() );

  //double idVMax = -1.0;
  //std::ssize_t iMax = -1;

  // Loop on isig, compute and sum for each abscissa the distance between isig (collected signal) and digit (first generated profile)
  for( std::size_t idc = 0; idc < rrBank.size(); ++idc )
  {
    // compute profile
    /// @todo to be pre-computed

    for( std::size_t i = 0; i < digit.size(); ++i )
    {
      const double xi = (i+1) * stepXi;
      std::ssize_t ldum = 0;
      for( std::size_t j = 0; j < rrBank[idc].size(); ++j )
      {
        if( 1.0 / rrBank[idc][j] <= xi )
        {
          ++ldum;
        }
      }
      BOOST_ASSERT( i < digit.size() );

      // set odd value to -1 and even value to 1
      digit[i] = - ( ldum % 2 ) * 2 + 1;
    }


    // compute distance to profile
    double d = 0;
    for( std::size_t i = startOffset; i < isig.size(); ++i )
    {
      d += dis( isig[i], digit[i], mub, muw, varSig );
    }

    const double v = std::exp( -d );

    sortedId[v] = idc;

  }

  int k = 0;
  BOOST_REVERSE_FOREACH( const MapT::const_iterator::value_type & v, sortedId )
  {
    if( k >= sizeIds ) break;
    std::pair< MarkerID, double > markerId;
    markerId.first = v.second;
    markerId.second = v.first;
    idSet.push_back(markerId);
    ++k;
  }

  return ( idSet.front().second > minIdentProba );
}

#ifdef USE_INITAL_REFINE_CONIC_FAMILY // unused, depreciated
bool refineConicFamily( CCTag & cctag, std::vector< cctag::ImageCut > & fsig, 
        const std::size_t lengthSig, const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2dN<double> > & pr,
        const bool useLmDif )
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;

  cctag::numerical::BoundedMatrix3x3d & mH = cctag.homography();
  Point2dN<double> & oRefined = cctag.centerImg();


  BOOST_ASSERT( pr.size() > 0 );

#ifdef WITH_CMINPACK
  if ( useLmDif )
  {
    DO_TALK( CCTAG_COUT_DEBUG( "Before optimizer: " << oRefined ); )
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    // Old lm optimization
    LMImageCenterOptimizer opt;
    opt( cctag );
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    DO_TALK( CCTAG_COUT_DEBUG( "After optimizer (LmDif) : " << oRefined << ", timer: " << spendTime ); )
  }
#else
  if ( useLmDif )
  {

    ImageCenterOptimizer opt( pr );

    CCTagVisualDebug::instance().newSession( "refineConicPts" );
    BOOST_FOREACH(const cctag::Point2dN<double> & pt, pr)
    {
      CCTagVisualDebug::instance().drawPoint( pt, cctag::color_red );
    }

    //oRefined = ellipse.center();

    CCTagVisualDebug::instance().newSession( "centerOpt" );
    CCTagVisualDebug::instance().drawPoint( oRefined, cctag::color_green );

    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

    // Optimization conditioning
    cctag::numerical::BoundedMatrix3x3d mT = cctag::numerical::optimization::conditionerFromPoints( pr );
    //cctag::numerical::BoundedMatrix3x3d mT = cctag::numerical::optimization::conditionerFromEllipse( ellipse );

    oRefined = opt( oRefined, lengthSig, src, ellipse, mT );

    // Check if the refined point is near the center of the outer ellipse.
    cctag::numerical::geometry::Ellipse semiEllipse( ellipse.center(),ellipse.a()/2.0,ellipse.b()/2.0,ellipse.angle() );
    if( !cctag::isInEllipse( semiEllipse, oRefined) )
      return false;

    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    DO_TALK( CCTAG_COUT_DEBUG( "After optimizer (optpp+interp2D) : " << oRefined << ", timer: " << spendTime ); )
  }
#endif
  else
  {

    //ImageCenterOptimizer opt( pr );

    CCTagVisualDebug::instance().newSession( "refineConicPts" );
    BOOST_FOREACH(const cctag::Point2dN<double> & pt, pr)
    {
      CCTagVisualDebug::instance().drawPoint( pt, cctag::color_red );
    }

    //oRefined = ellipse.center();

    CCTagVisualDebug::instance().newSession( "centerOpt" );
    CCTagVisualDebug::instance().drawPoint( oRefined, cctag::color_green );

    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );

    //oRefined = opt( oRefined, lengthSig, sourceView, ellipse.matrix() );

    // Optimization conditioning
    cctag::numerical::BoundedMatrix3x3d mT = cctag::numerical::optimization::conditionerFromPoints( pr );
    cctag::numerical::BoundedMatrix3x3d mInvT;
    cctag::numerical::invert_3x3(mT,mInvT);

    cctag::numerical::optimization::condition(oRefined, mT);
    /**********************************************************************/
    ceres::Problem problem;

    std::vector<double> x;
    x.push_back(oRefined.x());
    x.push_back(oRefined.y());

    ceres::CostFunction* cost_function =
      new ceres::NumericDiffCostFunction<TotoFunctor, ceres::CENTRAL, 1, 2> (new TotoFunctor(pr, lengthSig, src, ellipse, mT )   );
    problem.AddResidualBlock(cost_function, NULL, &x[0]);

    // Run the solver!
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::BFGS;
    //options.line_search_type = ceres::ARMIJO
    options.function_tolerance = 1.0e-4;
    //options.line_search_sufficient_curvature_decrease = 0.9; // Default.
    //options.numeric_derivative_relative_step_size = 1e-6;
    //options.max_num_iterations = 40;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    oRefined.setX(x[0]);
    oRefined.setY(x[1]);

    cctag::numerical::optimization::condition(oRefined, mInvT);
    /**********************************************************************/

    // Check if the refined point is near the center of the outer ellipse.
    cctag::numerical::geometry::Ellipse semiEllipse( ellipse.center(),ellipse.a()/2.0,ellipse.b()/2.0,ellipse.angle() );
    if( !cctag::isInEllipse( semiEllipse, oRefined) )
      return false;

    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    DO_TALK( CCTAG_COUT_DEBUG( "After optimizer (optpp+interp2D) : " << oRefined << ", timer: " << spendTime ); )
  }

  {
    // New optimization library...
    //ImageCenterOptimizer opt( pr );
  }

  CCTagVisualDebug::instance().drawPoint( oRefined, cctag::color_red );

  {
    //CCTAG_COUT_DEBUG( "Before getsignal" );
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    getSignals( mH, fsig, lengthSig, oRefined, pr, src, ellipse.matrix() );
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    //CCTAG_COUT_DEBUG( "After getsignal, timer: " << spendTime );
  }

  return true;
}
#endif // USE_INITAL_REFINE_CONIC_FAMILY // unused, depreciated

} // namespace identification
} // namespace cctag
