#include "params.hpp"
#include <boost/archive/archive_exception.hpp>

using namespace std;

namespace cctag
{
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
    , _debugDir( "" )
{
    _nCircles = 2*_nCrowns;
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

} // namespace cctag
