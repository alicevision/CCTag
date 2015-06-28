#include <cctag/identification.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/algebra/eig.hpp>
#include <cctag/algebra/invert.hpp>
#include <cctag/algebra/matrix/Matrix.hpp>
#include <cctag/algebra/matrix/operation.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/ImageCenterOptimizerCeres.hpp>

#include <terry/sampler/all.hpp>

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

#include <cmath>
#include <vector>

namespace cctag
{

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
  boost::numeric::ublas::vector<double> isig( cuts.front()._imgSignal.size() );
  BOOST_ASSERT( isig.size() - startOffset > 0 );

  // Sum all cuts to isig
  for( std::size_t i = 0; i < isig.size(); ++i )
  {
    double& isigCurrent = isig(i);
    isigCurrent = 0.0;
    BOOST_FOREACH( const cctag::ImageCut & cut, cuts )
    {
      isigCurrent += cut._imgSignal( i );
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

bool orazioDistanceRobust( std::vector<std::list<double> > & vScore,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        const std::size_t startOffset,
        const double minIdentProba,
        std::size_t sizeIds)
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

  BOOST_FOREACH( const cctag::ImageCut & cut, cuts )
  {
    MapT sortedId;

    std::size_t sizeIds = 6;
    IdSet idSet;
    idSet.reserve(sizeIds);

    // isig contains 1D signal on line.
    boost::numeric::ublas::vector<double> isig( cuts.front()._imgSignal.size() );
    BOOST_ASSERT( isig.size() - startOffset > 0 );

    // Sum all cuts to isig
    for( std::size_t i = 0; i < isig.size(); ++i )
    {
      double& isigCurrent = isig(i);
      isig(i) = cut._imgSignal( i );
    }

    // compute some statitics
    accumulator_set< double, features< /*tag::median,*/ tag::variance > > acc;
    // put sub signal into the statistical tool
    acc = std::for_each( isig.begin()+startOffset, isig.end(), acc );

    const double mSig = computeMedian( boost::numeric::ublas::subrange(isig,startOffset, isig.size()) );

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

    // find the nearest ID in rrBank
    const double stepXi = 1.0 / ( isig.size() + 1.0 ); /// @todo lilian +1 ??
    ///@todo vector<char>
    // vector of 1 or -1 values
    std::vector<double> digit( isig.size() );

#ifdef GRIFF_DEBUG
    assert( rrBank.size() > 0 );
#endif // GRIFF_DEBUG
    // Loop on isig, compute and sum for each abscissa the distance between 
    // isig (collected signal) and digit (first generated profile)
    for( std::size_t idc = 0; idc < rrBank.size(); ++idc )
    {
      // compute profile
      // @todo to be pre-computed

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

  //id = iMax;
  return true;//( idSet.front().second > minIdentProba );
}

cctag::numerical::BoundedMatrix3x3d adjustH( cctag::numerical::BoundedMatrix3x3d & mH,
    const cctag::Point2dN<double> & o,
    const cctag::Point2dN<double> & p )
{
  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;

  cctag::numerical::BoundedMatrix3x3d mInvH;

  invert( mH, mInvH );

  Point2dN<double> bo = prec_prod< BoundedVector3d >( mInvH, o );
  {
    BoundedMatrix3x3d mT;
    mT( 0, 0 ) = 1.0;
    mT( 0, 1 ) = 0.0;
    mT( 0, 2 ) = bo.x();
    mT( 1, 0 ) = 0.0;
    mT( 1, 1 ) =  1.0;
    mT( 1, 2 ) = bo.y();
    mT( 2, 0 ) = 0.0;
    mT( 2, 1 ) =  0.0;
    mT( 2, 2 ) = 1.0;

    mH = prec_prod( mH, mT );
    invert( mH, mInvH );
  }

  Point2dN<double> bp = prec_prod< BoundedVector3d >( mInvH, p );

  const double s = norm_2( subrange( bp, 0, 2 ) );

  BoundedVector3d d  = bp/s;

  {
    BoundedMatrix3x3d mT;
    mT( 0, 0 ) = s*d(0);
    mT( 0, 1 ) = -s*d(1);
    mT( 0, 2 ) = 0.0;
    mT( 1, 0 ) = s*d(1);
    mT( 1, 1 ) =  s*d(0);
    mT( 1, 2 ) = 0.0;
    mT( 2, 0 ) = 0.0;
    mT( 2, 1 ) =  0.0;
    mT( 2, 2 ) = 1.0;

    mH = prec_prod( mH, mT );
  }

  return mH;
}

void extractSignalUsingHomography( cctag::ImageCut & rectifiedSig,
        const cv::Mat & src,
        cctag::numerical::BoundedMatrix3x3d & mH,
        const std::size_t n, const double begin,
        const double end )
{
  using namespace boost;
  using namespace boost::numeric::ublas;
  using namespace cctag::numerical;

  BOOST_ASSERT( rectifiedSig._imgSignal.size() == 0 );
  BOOST_ASSERT( end >= begin );

  // Pour chaque coordonnees dans l image, on recupere le niveau de gris ZI(i), resultat de l'interpolation cubic2D
  // ( peut-etre peut-on passer la methode d'interpolation en param de la fonction, i.e. qu'on puisse appeler bilinear(linear2D) ou bicubic(cubic2D) )
  // au voisinage du point de coordonnees ( x = iPT(1,i), y = iPT(2,i) )

  const double stepXi = ( end - begin ) / ( n - 1.0 );
  rectifiedSig._imgSignal.resize( n );
  rectifiedSig._start = getHPoint( begin, 0.0, mH );
  rectifiedSig._stop = getHPoint( end, 0.0, mH );

  // Accumulator for mean value calculator (used when we are going outside the bounds)
  accumulators::accumulator_set< double, accumulators::features< accumulators::tag::mean > > acc;
  std::vector<std::size_t> idxNotInBounds;
  idxNotInBounds.reserve( n );
  for( std::size_t i = 0; i < n; ++i )
  {
    const double xi = i * stepXi + begin;
    const cctag::Point2dN<double> hp = getHPoint( xi, 0.0, mH );

    if ( hp.x() >= 1.0 && hp.x() <= src.cols-1 &&
         hp.y() >= 1.0 && hp.y() <= src.rows-1 )
    {
      // put pixel value to rectified signal
      double pixVal = getPixelBilinear( src, hp.x(), hp.y());
      rectifiedSig._imgSignal(i) = pixVal;
      acc( pixVal );
    }
    else
    {
      // push index
      idxNotInBounds.push_back( i );
    }
  }
  const double m = accumulators::mean( acc );
  BOOST_FOREACH( const std::size_t i, idxNotInBounds )
  {
    rectifiedSig._imgSignal(i) = m;
  }
}

std::size_t cutInterpolated( cctag::ImageCut & cut,
        const cv::Mat & src,
        const cctag::Point2dN<double> & pStart,
        const cctag::Point2dN<double> & pStop,
        const std::size_t nSteps )
{
  const double kx = ( pStop.x() - pStart.x() ) / ( nSteps - 1 );
  const double ky = ( pStop.y() - pStart.y() ) / ( nSteps - 1 );
  cut._imgSignal.resize( nSteps );
  cut._start = pStart;
  cut._stop = pStop;
  double x = pStart.x();
  double y = pStart.y();
  std::size_t len = 0;
  for( std::size_t i = 0; i < nSteps; ++i )
  {
    if ( x >= 1.0 && x < src.cols-1 &&
         y >= 1.0 && y < src.rows-1 )
    {
      // put pixel value to rectified signal
      cut._imgSignal(i) = double(getPixelBilinear( src, x, y));
      ++len;
    }
    else
    {
      // push black
      cut._imgSignal(i) = 0.0;
    }
    x += kx;
    y += ky;
  }
  return len;
}

void collectCuts( std::vector<cctag::ImageCut> & cuts,
        const cv::Mat & src,
        const cctag::Point2dN<double> & center,
        const std::vector< cctag::Point2dN<double> > & pts,
        const std::size_t sampleCutLength,
        const std::size_t startOffset )
{
  // collect signal from center to external ellipse point
  cuts.reserve( pts.size() );
  BOOST_FOREACH( const cctag::Point2dN<double> & p, pts )
  {
    cuts.push_back( cctag::ImageCut() );
    cctag::ImageCut & cut = cuts.back();
    if ( cutInterpolated( cut,
                          src,
                          center,
                          p,
                          sampleCutLength ) < ( sampleCutLength - startOffset ) )
    {
      cuts.pop_back();
    }
    ///@todo put a assert that checks that the collected cuts draw lines inside the image
  }
}

double costSelectCutFun( const std::vector<double> & varCuts,
        const boost::numeric::ublas::vector<std::size_t> & randomIdx,
        const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const double alpha)
{
  using namespace cctag::numerical;
  namespace ublas = boost::numeric::ublas;
  
  BoundedVector2d sumDeriv;
  double sumVar = 0;
  sumDeriv.clear();
  BOOST_FOREACH( const std::size_t i, randomIdx )
  {
    BOOST_ASSERT( i < varCuts.size() );

    ublas::bounded_vector<double,2> gradient;
    gradient(0) = dx.at<short>( collectedCuts[i]._stop.y(), collectedCuts[i]._stop.x() );
    gradient(1) = dy.at<short>( collectedCuts[i]._stop.y(), collectedCuts[i]._stop.x() );
    double normGrad = ublas::norm_2(gradient);
    sumDeriv(0) += gradient(0)/normGrad;
    sumDeriv(1) += gradient(1)/normGrad;
    sumVar += varCuts[i];

  }

  const double ndir = ublas::norm_2( sumDeriv );

  return ndir - alpha * sumVar;
}


void selectCut( std::vector< cctag::ImageCut > & cutSelection,
        std::vector< cctag::Point2dN<double> > & prSelection,
        std::size_t selectSize, const std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const cv::Mat & dx,
        const cv::Mat & dy,
        const double refinedSegSize,
        const std::size_t numSamplesOuterEdgePointsRefinement,
        const std::size_t cutsSelectionTrials )
{
  using namespace boost::numeric;
  using namespace boost::accumulators;

  selectSize = std::min( selectSize, collectedCuts.size() );

  std::vector<double> varCuts;
  varCuts.reserve( collectedCuts.size() );
  BOOST_FOREACH( const cctag::ImageCut & line, collectedCuts )
  {
    accumulator_set< double, features< tag::variance > > acc;
    acc = std::for_each( line._imgSignal.begin(), line._imgSignal.end(), acc );

    varCuts.push_back( variance( acc ) );
  }

  // On cherche nPT bons points parmi p0 qui maximisent la variance et minimise la norme de la somme des gradients normalisés
  ublas::vector<std::size_t> randomIdx = boost::numeric::ublas::subrange(
          cctag::numerical::randperm< ublas::vector<std::size_t> >( collectedCuts.size() ), 0, selectSize );
  double cost = costSelectCutFun( varCuts, randomIdx, collectedCuts, dx, dy );
  double Sm = cost;
  ublas::vector<std::size_t> idxSelected = randomIdx;

  ///@todo 2000 can be higher than the number of combinations, use std::min
  for( std::size_t i = 0; i < cutsSelectionTrials; ++i )
  {
    ublas::vector<std::size_t> randomIdx = boost::numeric::ublas::subrange( 
            cctag::numerical::randperm< ublas::vector<std::size_t> >( collectedCuts.size() ), 0, selectSize );
    
    double cost = costSelectCutFun( varCuts, randomIdx, collectedCuts, dx, dy );
    if ( cost < Sm )
    {
      Sm = cost;
      idxSelected = randomIdx;
    }
  }

  // Ordered map to get variance from the higher value to the lower
  typedef std::multimap< double, const cctag::ImageCut *, std::greater<double> > MapT;
  MapT mapVar;

  BOOST_FOREACH( const std::size_t i, idxSelected )
  {
    const cctag::ImageCut & line = collectedCuts[i];
    std::pair<double, const cctag::ImageCut*> v( varCuts[i], &line );
    mapVar.insert( v );
  }

  // half size of the segment used to refine the external point of cut
  const double halfWidth = refinedSegSize / 2.0;

  std::size_t i = 0;
  prSelection.reserve( selectSize );
  cutSelection.reserve( selectSize );
  BOOST_FOREACH( const MapT::value_type & v, mapVar )
  {
    const cctag::ImageCut & line = *v.second;
    BOOST_ASSERT( line._stop.x() >= 0 && line._stop.x() < dx.cols );
    BOOST_ASSERT( line._stop.y() >= 0 && line._stop.y() < dx.rows );
    BOOST_ASSERT( line._stop.x() >= 0 && line._stop.x() < dy.cols );
    BOOST_ASSERT( line._stop.y() >= 0 && line._stop.y() < dy.rows );

    cctag::numerical::BoundedVector3d gradDirection;
    gradDirection( 0 ) = dx.at<short>(line._stop.y(), line._stop.x());
    gradDirection( 1 ) = dy.at<short>(line._stop.y(), line._stop.x());
    gradDirection( 2 ) = 0.0;
    gradDirection = cctag::numerical::unit( gradDirection );

    BOOST_ASSERT( norm_2( gradDirection ) != 0 );

    const Point2dN<double> start( line._stop - halfWidth * gradDirection );
    const Point2dN<double> stop( line._stop + halfWidth * gradDirection );

    // collect signal from e1 to e2
    cctag::ImageCut cut;
    cutInterpolated(
      cut,
      src,
      start,
      stop,
      numSamplesOuterEdgePointsRefinement );

    //CCTAG_TCOUT_VAR( line._stop ); //don't delete.

    //SubPixEdgeOptimizer optimizer( cut );

    cctag::Point2dN<double> refinedPoint(line._stop);
    //cctag::Point2dN<double> refinedPoint = optimizer( halfWidth, line._stop.x(), cut._imgSignal[0], cut._imgSignal[ cut._imgSignal.size() - 1 ] );


    //CCTAG_TCOUT_VAR( refinedPoint ); //don't delete.
    // Take cuts the didn't diverge too much
    if ( cctag::numerical::distancePoints2D( line._stop, refinedPoint ) < halfWidth )
    {

      prSelection.push_back( refinedPoint );
      //prSelection.push_back( line._stop );//!!!!!!!!!!!!!!!!!!!!!!!!!!!

      cutSelection.push_back( line );
    }
    ++i;
    if( cutSelection.size() >= selectSize )
    {
      break;
    }
  }
}

bool getSignals( cctag::numerical::BoundedMatrix3x3d & mH,
        std::vector< cctag::ImageCut > & signals,
        const std::size_t lengthSig, const cctag::Point2dN<double> & o,
        const std::vector< cctag::Point2dN<double> > & vecExtPoint, 
        const cv::Mat & src, 
        const cctag::numerical::BoundedMatrix3x3d & matEllipse )
{
  BOOST_ASSERT( vecExtPoint.size() > 0 );

  // Check if we aren't diverging
  if( o.x() < -150 || o.x() > src.cols+150 || o.y() < -150 || o.y() > src.rows+150 )
    return false;

  using namespace cctag::numerical;
  using namespace boost::numeric::ublas;

  //computeHomographyFromEllipseAndImagedCenter(matEllipse, mH ....); todo@Lilian

  ///@todo eloi Begin => to function
  cctag::numerical::BoundedMatrix3x3d mA;
  invert( matEllipse, mA );
  cctag::numerical::BoundedMatrix3x3d mO = outer_prod( o, o );
  diagonal_matrix<double> vpg;

  cctag::numerical::BoundedMatrix3x3d mVG;
  // Compute eig(inv(A),o*o')
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

  column( mH, 0 ) = column( mU, 0 );
  column( mH, 1 ) = column( mU, 1 );
  column( mH, 2 ) = cross( column( mU, 0 ), column( mU, 1 ) );

  // end : homographyFromOuterEllipseCenter(MAtrix3x3d & mH, const MAtrix3x3d & matEllipse, bounded_vector<double,3> o)

  mH = adjustH( mH, o, vecExtPoint.front() );

  cctag::numerical::normalizeDet1( mH );

  // We use a temporary matrix mHrot to rotate around the ellipse and gets imagecut signals.
  cctag::numerical::BoundedMatrix3x3d mHrot = mH;

  signals.resize( vecExtPoint.size() );
  // First pass
  {
    cctag::ImageCut & rectifiedSig = signals.front();
    extractSignalUsingHomography( rectifiedSig, src, mHrot, lengthSig );
  }


  for( std::size_t i = 1; i < vecExtPoint.size(); ++i )
  {
    mHrot = adjustH( mHrot, o, vecExtPoint[i] );
    cctag::ImageCut & rectifiedSig = signals[i];
    extractSignalUsingHomography( rectifiedSig, src, mHrot, lengthSig );
  }
  return true;
}


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
    CCTAG_COUT_DEBUG( "Before optimizer: " << oRefined );
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    // Old lm optimization
    LMImageCenterOptimizer opt;
    opt( cctag );
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    CCTAG_COUT_DEBUG( "After optimizer (LmDif) : " << oRefined << ", timer: " << spendTime );
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
    CCTAG_COUT_DEBUG( "After optimizer (optpp+interp2D) : " << oRefined << ", timer: " << spendTime );
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
    CCTAG_COUT_DEBUG( "After optimizer (optpp+interp2D) : " << oRefined << ", timer: " << spendTime );
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

int identify(
  CCTag & cctag,
  const std::vector< std::vector<double> > & radiusRatios, ///@todo directly use the bank
  const cv::Mat & src,
  const cv::Mat & dx,
  const cv::Mat & dy,
  const cctag::Parameters & params)
{
  const cctag::numerical::geometry::Ellipse & ellipse = cctag.rescaledOuterEllipse();
  const std::vector< cctag::Point2dN<double> > & outerEllipsePoints = cctag.rescaledOuterEllipsePoints();
  // outerEllipsePoints can be changed in the edge point refinement - not const - todo@Lilian - save their modifications
  // in the CCTag instance just above _rescaledOuterEllipsePoints.

  // Take 50 edge points around outer ellipse.
  const std::size_t n = std::min( std::size_t(100), outerEllipsePoints.size() );//50
  std::size_t step = std::size_t( outerEllipsePoints.size() / ( n - 1 ) );
  std::vector< cctag::Point2dN<double> > ellipsePoints;

  ellipsePoints.reserve( n );
  for( std::size_t i = 0; i < outerEllipsePoints.size(); i += step )
  {
    ellipsePoints.push_back( outerEllipsePoints[ i ] );
  }

  ///@todo Check if a & b sorted (cf. ellipse2param)
  const double refinedSegSize = std::min( ellipse.a(), ellipse.b() ) * 0.12;

  BOOST_FOREACH(const cctag::Point2dN<double> & pt, ellipsePoints)
  {
    CCTagVisualDebug::instance().drawPoint( pt, cctag::color_green );
  }

  std::size_t startOffset = 0;

  if (params._numCrowns == 3)
  {
    // Signal begin at 25% of the radius (for 3 crowns markers).
    startOffset = params._sampleCutLength-(2*params._numCrowns-1)*0.15*params._sampleCutLength;	// Considering 6 radius.
  }
  else if (params._numCrowns == 4)
  {
    startOffset = 26;
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
    //CCTAG_TCOUT( "Before collectCuts" );
    // Collect cuts around the extern ellipse with the interval [startOffset;1.0] not outside the image
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    // Collect cuts around the extern ellipse with the interval [startOffset;1.0] not outside the image
    collectCuts( cuts, src, ellipse.center(), ellipsePoints, params._sampleCutLength, startOffset);
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
    //CCTAG_TCOUT( "After collectCuts, timer: " << spendTime );
    //CCTAG_TCOUT_VAR( cuts.size() );
  }
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t1(boost::posix_time::microsec_clock::local_time());

  boost::posix_time::time_duration d = t1 - t0;
  double spendTime = d.total_milliseconds();
  CCTAG_COUT_OPTIM("Time in collectCuts: " << spendTime << " ms");
#endif

  if ( cuts.size() == 0 )
  {
    // Can happen when an object or the image frame is occluding a part of all available cuts.
    return status::no_collected_cuts;
  }
  
  std::vector< cctag::ImageCut > cutSelection;
  std::vector< cctag::Point2dN<double> > prSelection;

  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    
    selectCut( cutSelection, prSelection, params._numCutsInIdentStep, cuts, src, 
            dx, dy, refinedSegSize, params._numSamplesOuterEdgePointsRefinement,
            params._cutsSelectionTrials );
    
    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
  }
  
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t2(boost::posix_time::microsec_clock::local_time());

  d = t2 - t1;
  spendTime = d.total_milliseconds();
  CCTAG_COUT_OPTIM("Time in selectCut: " << spendTime << " ms");
#endif

  if ( prSelection.size() == 0 )
  {
    // Can happen when the refined edge point estimation is diverging for all cuts.
    return status::no_selected_cuts;
  }

  std::vector< cctag::ImageCut > fsig;

  {
    boost::posix_time::ptime tstart( boost::posix_time::microsec_clock::local_time() );
    bool hasConverged = refineConicFamily( cctag, fsig, params._sampleCutLength, src, ellipse, prSelection, params._useLMDif );
    if( !hasConverged )
    {
      CCTAG_COUT_DEBUG(ellipse);
      CCTAG_COUT_VAR_DEBUG(cctag.centerImg());
      CCTAG_COUT_DEBUG( "Optimization on imaged center failed to converge." );
      return status::opti_has_diverged;
    }
    
#ifdef CCTAG_OPTIM
  boost::posix_time::ptime t3(boost::posix_time::microsec_clock::local_time());

  d = t3 - t2;
  spendTime = d.total_milliseconds();
  CCTAG_COUT_OPTIM("Time in refineConicFamily: " << spendTime << " ms");
#endif

    boost::posix_time::ptime tend( boost::posix_time::microsec_clock::local_time() );
    boost::posix_time::time_duration d = tend - tstart;
    const double spendTime = d.total_milliseconds();
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

    if(0)
    {
      idFinal = orazioDistance( idSet, radiusRatios, fsig, startOffset, params._minIdentProba, sizeIds);
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
        CCTAG_COUT_DEBUG("Not enough quality in IDENTIFICATION");
      }
    }
    else
    {
      idFinal = orazioDistanceRobust( vScore, radiusRatios, fsig, startOffset, params._minIdentProba, sizeIds);
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

        idFinal = (score > params._minIdentProba);
#ifdef GRIFF_DEBUG
      }
#endif // GRIFF_DEBUG
    }

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

} // namespace cctag
