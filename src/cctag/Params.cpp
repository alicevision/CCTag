/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "Params.hpp"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <boost/archive/xml_iarchive.hpp>

namespace cctag
{

bool Parameters::OverrideChecked = false;
bool Parameters::OverrideLoaded = false;
Parameters Parameters::Override;

void Parameters::LoadOverride()
{
  const char* path = getenv("CCTAG_PARAMETERS_OVERRIDE");
  if (!path) path = "./CCTagParametersOverride.xml";
  std::ifstream ifs(path);
  if (!ifs)
    return;
  
  boost::archive::xml_iarchive ia(ifs);
  ia >> boost::serialization::make_nvp("CCTagsParams", Override);
  OverrideLoaded = true;
  std::cout << "CCTag: loaded parameters override file: " << path << std::endl;
}

Parameters::Parameters(const std::size_t nCrowns)
    : _cannyThrLow( kDefaultCannyThrLow )
    , _cannyThrHigh( kDefaultCannyThrHigh )
    , _distSearch( kDefaultDistSearch )
    , _thrGradientMagInVote( kDefaultThrGradientMagInVote )
    , _angleVoting( kDefaultAngleVoting )
    , _ratioVoting( kDefaultRatioVoting )
    , _averageVoteMin( kDefaultAverageVoteMin )
    , _thrMedianDistanceEllipse( kDefaultThrMedianDistanceEllipse )
    , _maximumNbSeeds( kDefaultMaximumNbSeeds )
    , _maximumNbCandidatesLoopTwo( kDefaultMaximumNbCandidatesLoopTwo )
    , _nCrowns( nCrowns )
    , _minPointsSegmentCandidate( kDefaultMinPointsSegmentCandidate )
    , _minVotesToSelectCandidate( kDefaultMinVotesToSelectCandidate )
    , _threshRobustEstimationOfOuterEllipse( kDefaultThreshRobustEstimationOfOuterEllipse )
    , _ellipseGrowingEllipticHullWidth( kDefaultEllipseGrowingEllipticHullWidth )
    , _windowSizeOnInnerEllipticSegment( kDefaultWindowSizeOnInnerEllipticSegment )
    , _numberOfMultiresLayers( kDefaultNumberOfMultiresLayers )
    , _nSamplesOuterEllipse( kDefaultNSamplesOuterEllipse )
    , _numberOfProcessedMultiresLayers( kDefaultNumberOfProcessedMultiresLayers )
    , _numCutsInIdentStep( kDefaultNumCutsInIdentStep )
    , _numSamplesOuterEdgePointsRefinement( kDefaultNumSamplesOuterEdgePointsRefinement )
    , _cutsSelectionTrials( kDefaultCutsSelectionTrials )
    , _sampleCutLength( kDefaultSampleCutLength )
    , _imagedCenterNGridSample( kDefaultImagedCenterNGridSample )
    , _imagedCenterNeighbourSize( kDefaultImagedCenterNeighbourSize )
    , _minIdentProba( kDefaultMinIdentProba )
    , _useLMDif( kDefaultUseLMDif )
    , _searchForAnotherSegment( kDefaultSearchForAnotherSegment )
    , _writeOutput( kDefaultWriteOutput )
    , _doIdentification( kDefaultDoIdentification )
    , _maxEdges( kDefaultMaxEdges )
    , _useCuda( kDefaultUseCuda )
    , _debugDir( "" )
{
    _nCircles = 2*_nCrowns;
    
    if (!OverrideChecked) {
      OverrideChecked = true;
      LoadOverride();
    }
}

void Parameters::setDebugDir( const std::string& debugDir )
{
    struct stat st = {0};

    std::string dir = debugDir;
    char   dirtail = dir[ dir.size()-1 ];
    if( dirtail != '/' ) {
        _debugDir = debugDir + "/";
    } else {
        _debugDir = debugDir;
    }

    if (::stat( _debugDir.c_str(), &st) == -1) {
        ::mkdir( _debugDir.c_str(), 0700);
    }
}

void Parameters::setUseCuda( bool val )
{
#ifdef WITH_CUDA
    _useCuda = val;
#else
    if(val)
        std::cerr << "Warning: CCTag library is built without CUDA support, so we can't enable CUDA." << std::endl;
#endif // WITH_CUDA
}

} // namespace cctag
