/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <boost/math/constants/constants.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <sys/stat.h>  // needed for stat and mkdir
#include <sys/types.h> // needed for stat and mkdir

#define NO_WEIGHT 0
#define INV_GRAD_WEIGHT 1
#define INV_SQRT_GRAD_WEIGHT 2
#define INV_SQUARE_GRAD_WEIGHT 3

namespace cctag
{

static const std::size_t kDefaultDistSearch = 30;
static const std::size_t kDefaultNCrowns  = 3;//4;
static const std::size_t kDefaultNCircles = 6;//8;
static const int kDefaultThrGradientMagInVote = 2500;
static const float kDefaultAngleVoting        = 0.0f;
static const float kDefaultRatioVoting        = 4.f;
static const float kDefaultAverageVoteMin        = 0.f;
static const float kDefaultThrMedianDistanceEllipse = 3.0f;
static const std::size_t kDefaultMaximumNbSeeds = 500;
static const std::size_t kDefaultMaximumNbCandidatesLoopTwo = 40;//30;
static const float kDefaultCannyThrLow      =  0.01f ;//0.002
static const float kDefaultCannyThrHigh     =  0.04f ;// 0.04
static const std::size_t kDefaultMinPointsSegmentCandidate =  10;
static const std::size_t kDefaultMinVotesToSelectCandidate =  3;
static const float kDefaultThreshRobustEstimationOfOuterEllipse =  30.0f;
static const float kDefaultEllipseGrowingEllipticHullWidth =  2.3f;
static const std::size_t kDefaultWindowSizeOnInnerEllipticSegment =  20;
static const std::size_t kDefaultNumberOfMultiresLayers = 4;
static const std::size_t kDefaultNumberOfProcessedMultiresLayers = 4;
static const std::size_t kDefaultNSamplesOuterEllipse = 150;
static const std::size_t kDefaultNumCutsInIdentStep = 22;//30;//100;//15;
static const std::size_t kDefaultNumSamplesOuterEdgePointsRefinement = 20;
static const std::size_t kDefaultCutsSelectionTrials = 500;//10000;
static const std::size_t kDefaultSampleCutLength = 100;
static const std::size_t kDefaultImagedCenterNGridSample = 5; // must be odd otherwise the ellipse center will not be included 
                                                              // in the nearby points.
static const float kDefaultImagedCenterNeighbourSize = 0.20f;
static const float kDefaultMinIdentProba = 1e-6f;//1e-6
static const bool kDefaultUseLMDif = true;
static const bool kDefaultSearchForAnotherSegment = true;
static const bool kDefaultWriteOutput = false;
static const bool kDefaultDoIdentification = true;
static const uint32_t kDefaultMaxEdges = 20000;
#ifdef WITH_CUDA
static const bool kDefaultUseCuda = true;
#else
static const bool kDefaultUseCuda = false;
#endif

static const std::string kParamCannyThrLow( "kParamCannyThrLow" );
static const std::string kParamCannyThrHigh( "kParamCannyThrHigh" );
static const std::string kParamDistSearch( "kParamDistSearch" );
static const std::string kThrGradientMagInVote("kThrGradientMagInVote");
static const std::string kParamAngleVoting( "kParamAngleVoting" );
static const std::string kParamRatioVoting( "kParamRatioVoting" );
static const std::string kParamAverageVoteMin( "kParamAverageVoteMin" );
static const std::string kParamThrMedianDistanceEllipse( "kParamThrMedianDistanceEllipse" );
static const std::string kParamMaximumNbSeeds( "kParamMaximumNbSeeds" );
static const std::string kParamMaximumNbCandidatesLoopTwo( "kParamMaximumNbCandidatesLoopTwo" );
static const std::string kParamNCrowns( "kParamNCrowns" );
static const std::string kParamMinPointsSegmentCandidate( "kParamMinPointsSegmentCandidate" );
static const std::string kParamMinVotesToSelectCandidate( "kParamMinVotesToSelectCandidate" );
static const std::string kParamThreshRobustEstimationOfOuterEllipse( "kParamThreshRobustEstimationOfOuterEllipse" );
static const std::string kParamEllipseGrowingEllipticHullWidth( "kParamEllipseGrowingEllipticHullWidth" );
static const std::string kParamWindowSizeOnInnerEllipticSegment( "kParamWindowSizeOnInnerEllipticSegment" );
static const std::string kParamNumberOfMultiresLayers( "kParamNumberOfMultiresLayers" );
static const std::string kParamNumberOfProcessedMultiresLayers( "kParamNumberOfProcessedMultiresLayers" );
static const std::string kParamNSamplesOuterEllipse( "kParamNSamplesOuterEllipse" );
static const std::string kParamNumCutsInIdentStep( "kParamNumCutsInIdentStep" );
static const std::string kParamNumSamplesOuterEdgePointsRefinement( "kParamNumSamplesOuterEdgePointsRefinement" );
static const std::string kParamCutsSelectionTrials( "kParamCutsSelectionTrials" );
static const std::string kParamSampleCutLength( "kParamSampleCutLength" );
static const std::string kParamImagedCenterNGridSample( "kParamImagedCenterNGridSample" );
static const std::string kParamImagedCenterNeighbourSize( "kParamImagedCenterNeighbourSize" );
static const std::string kParamMinIdentProba( "kParamMinIdentProba" );
static const std::string kParamUseLMDif( "kParamUseLMDif" );
static const std::string kParamSearchForAnotherSegment( "kParamSearchForAnotherSegment" );
static const std::string kParamWriteOutput( "kParamWriteOutput" );
static const std::string kParamDoIdentification( "kParamDoIdentification" );
static const std::string kParamMaxEdges( "kParamMaxEdges" );
static const std::string kUseCuda( "kUseCuda" );

static const std::size_t kWeight = INV_GRAD_WEIGHT;

struct Parameters
{
  friend class boost::serialization::access;
  
  static bool OverrideChecked;
  static bool OverrideLoaded;
  static Parameters Override;
  static void LoadOverride();

  explicit Parameters(std::size_t nCrowns = kDefaultNCrowns);

  float _cannyThrLow; // canny low threshold
  float _cannyThrHigh; // canny high threshold
  std::size_t _distSearch; // maximum distance (in pixels) of research from one edge points
  // to another one. maximum length of a arc segment composing the polygonal line.
  int _thrGradientMagInVote; // during the voting procedure, the gradient direction
  // is followed as long as the gradient magnitude is
  // greater than this threshold
  float _angleVoting; // maximum angle between of gradient directions of two consecutive
  // edge points.
  float _ratioVoting; // maximum distance ratio between of gradient directions of two consecutive
  // edge points.
  float _averageVoteMin;
  float _thrMedianDistanceEllipse;
  std::size_t _maximumNbSeeds; // number of seeds to process as potential candidates
  std::size_t _maximumNbCandidatesLoopTwo;
  std::size_t _nCrowns; // number of crowns
  std::size_t _nCircles; // number of circles
  std::size_t _minPointsSegmentCandidate; // minimal number of points on the outer ellipse to select an inner segment candidate
  std::size_t _minVotesToSelectCandidate; // minimum number of received votes to select an edge
  // point as a new seed.
  float _threshRobustEstimationOfOuterEllipse; // LMeDs threshold on robust estimation of the outer ellipse
  float _ellipseGrowingEllipticHullWidth; // width of elliptic hull in ellipse growing
  std::size_t _windowSizeOnInnerEllipticSegment; // window size on the inner elliptic segment
  std::size_t _numberOfMultiresLayers; // number of multi-resolution layers
  std::size_t _numberOfProcessedMultiresLayers; // number of processed layers in multi-resolution
  std::size_t _nSamplesOuterEllipse; // Number of points considered in the preliminary sampling in identification
  std::size_t _numCutsInIdentStep; // number of cuts in the identification step
  std::size_t _numSamplesOuterEdgePointsRefinement; // number of sample for the outer edge points refinement in identification
  std::size_t _cutsSelectionTrials; // number of trials in cuts selection
  std::size_t _sampleCutLength; // sample cut length
  std::size_t  _imagedCenterNGridSample; // number of point sample along the square grid side (e.g. 5 for a grid of 5x5=25 points)
  float _imagedCenterNeighbourSize; // grid width relatively to the length of longest semi-axis of the outer ellipse
  float _minIdentProba; // minimal probability of delivered by the identification algorithm
  // to consider a candidate as a CCTag
  bool _useLMDif;
  bool _searchForAnotherSegment; // is CCTag can be made of many flow components (2).
  // This implies an assembling step which may be time
  // time consuming.
  bool _writeOutput;
  bool _doIdentification; // perform the identification step
  uint32_t _maxEdges; // max number of edge point, determines memory allocation
  bool        _useCuda; // if compiled WITH_CUDA, allow CLI selection, ignore if not
  std::string _debugDir; // prefix for debug output !!!! ONLY ON COMMAND LINE

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP( _cannyThrLow );
    ar & BOOST_SERIALIZATION_NVP( _cannyThrHigh );
    ar & BOOST_SERIALIZATION_NVP( _distSearch );
    ar & BOOST_SERIALIZATION_NVP( _thrGradientMagInVote );
    ar & BOOST_SERIALIZATION_NVP( _angleVoting );
    ar & BOOST_SERIALIZATION_NVP( _ratioVoting );
    ar & BOOST_SERIALIZATION_NVP( _averageVoteMin);
    ar & BOOST_SERIALIZATION_NVP( _thrMedianDistanceEllipse);
    ar & BOOST_SERIALIZATION_NVP( _maximumNbSeeds);
    ar & BOOST_SERIALIZATION_NVP( _maximumNbCandidatesLoopTwo);
    ar & BOOST_SERIALIZATION_NVP( _nCrowns );
    ar & BOOST_SERIALIZATION_NVP( _minPointsSegmentCandidate );
    ar & BOOST_SERIALIZATION_NVP( _minVotesToSelectCandidate );
    ar & BOOST_SERIALIZATION_NVP( _threshRobustEstimationOfOuterEllipse );
    ar & BOOST_SERIALIZATION_NVP( _ellipseGrowingEllipticHullWidth );
    ar & BOOST_SERIALIZATION_NVP( _windowSizeOnInnerEllipticSegment );
    ar & BOOST_SERIALIZATION_NVP( _numberOfMultiresLayers );
    ar & BOOST_SERIALIZATION_NVP( _numberOfProcessedMultiresLayers );
    ar & BOOST_SERIALIZATION_NVP( _nSamplesOuterEllipse );
    ar & BOOST_SERIALIZATION_NVP( _numCutsInIdentStep );
    ar & BOOST_SERIALIZATION_NVP( _numSamplesOuterEdgePointsRefinement );
    ar & BOOST_SERIALIZATION_NVP( _cutsSelectionTrials );
    ar & BOOST_SERIALIZATION_NVP( _sampleCutLength );
    ar & BOOST_SERIALIZATION_NVP( _imagedCenterNGridSample );
    ar & BOOST_SERIALIZATION_NVP( _imagedCenterNeighbourSize );
    ar & BOOST_SERIALIZATION_NVP( _minIdentProba );
    ar & BOOST_SERIALIZATION_NVP( _useLMDif );
    ar & BOOST_SERIALIZATION_NVP( _searchForAnotherSegment );
    ar & BOOST_SERIALIZATION_NVP( _writeOutput );
    ar & BOOST_SERIALIZATION_NVP( _doIdentification );
    ar & BOOST_SERIALIZATION_NVP( _maxEdges );
    ar & BOOST_SERIALIZATION_NVP( _useCuda );
    _nCircles = 2*_nCrowns;
  }

  void setDebugDir( const std::string& debugDir );

  void setUseCuda( bool val );
};

} // namespace cctag
