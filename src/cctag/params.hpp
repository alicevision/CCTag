#ifndef _ROM_VISION_CCTAG_PARAMS_HPP_
#define _ROM_VISION_CCTAG_PARAMS_HPP_

#include <boost/math/constants/constants.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

#include <cmath>
#include <cstddef>
#include <string>

//#include <rom/engine/RomConfig.hpp>

#define NO_WEIGHT 0
#define INV_GRAD_WEIGHT 1
#define INV_SQRT_GRAD_WEIGHT 2

namespace rom {
namespace vision {
namespace marker {
namespace cctag {

static const std::size_t kDefaultDistSearch = 30;
static const std::size_t kDefaultNumCrowns  = 3; // TODO to delete ~= _numCircles/2
static const std::size_t kDefaultNumCircles = 6;
static const float kDefaultAngleVoting        = 0.f;//std::cos( boost::math::constants::pi<float>() / 3.0 );
static const float kDefaultRatioVoting        = 4.f;//3.f;     // @todo: check 2 as default
static const float kDefaultAverageVoteMin        = 1.2f;
static const double kDefaultThrMedianDistanceEllipse = 3.0; //Default for 4 crowns.
static const std::size_t kDefaultMaximumNbSeeds = 40;
static const std::size_t kDefaultMaximumNbCandidatesLoopTwo = 40;
//static const float kDefaultCannyThrLow      =  0.025f; // Canny low threshold
//static const float kDefaultCannyThrHigh     =  0.1f; // Canny high threshold
static const float kDefaultCannyThrLow      =  0.2f ; // Canny low threshold
static const float kDefaultCannyThrHigh     =  0.4f ; // Canny high threshold
static const std::size_t kDefaultMinPointsSegmentCandidate =  15; // Minimal number of points on the outer ellipse to select an inner segment candidate
static const std::size_t kDefaultMinVotesToSelectCandidate =  4;  // Minimal number of received votes to select a candidate point
static const double kDefaultThreshRobustEstimationOfOuterEllipse =  40.0;  // LMeDs threshold on robust estimation of the outer ellipse
static const double kDefaultEllipseGrowingEllipticHullWidth =  3.0;  // Width of elliptic hull in ellipse growing
static const std::size_t kDefaultWindowSizeOnInnerEllipticSegment =  20;  // Window size on the inner elliptic segment
static const std::size_t kDefaultNumberOfMultiresLayers = 3;  // Number of multiresolution layers
static const std::size_t kDefaultNumberOfProcessedMultiresLayers = 3;  // Number of processed layers in multiresolution
static const std::size_t kDefaultNumCutsInIdentStep = 12;  // Number of cuts in the identification step
static const std::size_t kDefaultNumSamplesOuterEdgePointsRefinement = 10;  // Number of sample for the outer edge points refinement in identification
static const std::size_t kDefaultCutsSelectionTrials = 2000;  // Number of trials in cuts selection
static const std::size_t kDefaultSampleCutLength = 100;  // Sample cut length
static const double kDefaultMinIdentProba = 1e-14;  // Minimal probability of identification
static const bool kDefaultUseLMDif = false;  // Use old method
static const bool kDefaultSearchForAnotherSegment = true;
static const bool kDefaultWriteOutput = false;

static const std::string kParamCCTagBankFilename( "kParamCCTagBankFilename" );
static const std::string kParamCannyThrLow( "kParamCannyThrLow" );
static const std::string kParamCannyThrHigh( "kParamCannyThrHigh" );
static const std::string kParamDistSearch( "kParamDistSearch" );
static const std::string kParamAngleVoting( "kParamAngleVoting" );
static const std::string kParamRatioVoting( "kParamRatioVoting" );
static const std::string kParamAverageVoteMin( "kParamAverageVoteMin" );
static const std::string kParamThrMedianDistanceEllipse( "kParamThrMedianDistanceEllipse" );
static const std::string kParamMaximumNbSeeds( "kParamMaximumNbSeeds" );
static const std::string kParamMaximumNbCandidatesLoopTwo( "kParamMaximumNbCandidatesLoopTwo" );
static const std::string kParamNumCrowns( "kParamNumCrowns" );
static const std::string kParamMinPointsSegmentCandidate( "kParamMinPointsSegmentCandidate" );
static const std::string kParamMinVotesToSelectCandidate( "kParamMinVotesToSelectCandidate" );
static const std::string kParamThreshRobustEstimationOfOuterEllipse( "kParamThreshRobustEstimationOfOuterEllipse" );
static const std::string kParamEllipseGrowingEllipticHullWidth( "kParamEllipseGrowingEllipticHullWidth" );
static const std::string kParamWindowSizeOnInnerEllipticSegment( "kParamWindowSizeOnInnerEllipticSegment" );
static const std::string kParamNumberOfMultiresLayers( "kParamNumberOfMultiresLayers" );
static const std::string kParamNumberOfProcessedMultiresLayers( "kParamNumberOfProcessedMultiresLayers" );
static const std::string kParamNumCutsInIdentStep( "kParamNumCutsInIdentStep" );
static const std::string kParamNumSamplesOuterEdgePointsRefinement( "kParamNumSamplesOuterEdgePointsRefinement" );
static const std::string kParamCutsSelectionTrials( "kParamCutsSelectionTrials" );
static const std::string kParamSampleCutLength( "kParamSampleCutLength" );
static const std::string kParamMinIdentProba( "kParamMinIdentProba" );
static const std::string kParamUseLMDif( "kParamUseLMDif" );
static const std::string kParamSearchForAnotherSegment( "kParamSearchForAnotherSegment" );
static const std::string kParamWriteOutput( "kParamWriteOutput" );

static const std::size_t kWeight = INV_GRAD_WEIGHT;

struct Parameters
{
    friend class boost::serialization::access;
    Parameters()
    : _cctagBankFilename("TODO")
    , _cannyThrLow( cctag::kDefaultCannyThrLow )
    , _cannyThrHigh( cctag::kDefaultCannyThrHigh )
    , _distSearch( cctag::kDefaultDistSearch )
    , _angleVoting( cctag::kDefaultAngleVoting )
    , _ratioVoting( cctag::kDefaultRatioVoting )
    , _averageVoteMin( cctag::kDefaultAverageVoteMin )
    , _thrMedianDistanceEllipse( cctag::kDefaultThrMedianDistanceEllipse )
    , _maximumNbSeeds( cctag::kDefaultMaximumNbSeeds ) 
    , _maximumNbCandidatesLoopTwo( cctag::kDefaultMaximumNbCandidatesLoopTwo )
    , _numCrowns( cctag::kDefaultNumCrowns )
    , _minPointsSegmentCandidate( cctag::kDefaultMinPointsSegmentCandidate )
    , _minVotesToSelectCandidate( cctag::kDefaultMinVotesToSelectCandidate )
    , _threshRobustEstimationOfOuterEllipse( cctag::kDefaultThreshRobustEstimationOfOuterEllipse )
    , _ellipseGrowingEllipticHullWidth( cctag::kDefaultEllipseGrowingEllipticHullWidth )
    , _windowSizeOnInnerEllipticSegment( cctag::kDefaultWindowSizeOnInnerEllipticSegment )
    , _numberOfMultiresLayers( cctag::kDefaultNumberOfMultiresLayers )
    , _numberOfProcessedMultiresLayers( cctag::kDefaultNumberOfProcessedMultiresLayers )
    , _numCutsInIdentStep( cctag::kDefaultNumCutsInIdentStep )
    , _numSamplesOuterEdgePointsRefinement( cctag::kDefaultNumSamplesOuterEdgePointsRefinement )
    , _cutsSelectionTrials( cctag::kDefaultCutsSelectionTrials )
    , _sampleCutLength( cctag::kDefaultSampleCutLength )
    , _minIdentProba( cctag::kDefaultMinIdentProba )
    , _useLMDif( cctag::kDefaultUseLMDif )
    , _searchForAnotherSegment( cctag::kDefaultSearchForAnotherSegment )
    , _writeOutput( cctag::kDefaultWriteOutput )
    {
        _nCircles = 2*_numCrowns;
    }

    std::string _cctagBankFilename;
    float _cannyThrLow;
    float _cannyThrHigh;
    std::size_t _distSearch; // maximum distance (in pixels) of research from one edge points
                             // to another one. maximum length of a arc segment composing the polygonal line.
    float _angleVoting; // maximum angle between of gradient directions of two consecutive
                        // edge points.
    float _ratioVoting; // maximum distance ratio between of gradient directions of two consecutive
                        // edge points.
    float _averageVoteMin;
    double _thrMedianDistanceEllipse;
    std::size_t _maximumNbSeeds;
    std::size_t _maximumNbCandidatesLoopTwo;
    std::size_t _numCrowns;
    std::size_t _nCircles;
    std::size_t _minPointsSegmentCandidate;
    std::size_t _minVotesToSelectCandidate; // minimum number of received votes to select an edge 
                                            // point as a new seed.
    double _threshRobustEstimationOfOuterEllipse;
    double _ellipseGrowingEllipticHullWidth;
    std::size_t _windowSizeOnInnerEllipticSegment;
    std::size_t _numberOfMultiresLayers;
    std::size_t _numberOfProcessedMultiresLayers;
    std::size_t _numCutsInIdentStep;
    std::size_t _numSamplesOuterEdgePointsRefinement;
    std::size_t _cutsSelectionTrials;
    std::size_t _sampleCutLength;
    double _minIdentProba;
    bool _useLMDif;
    bool _searchForAnotherSegment;
    bool _writeOutput;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP( _cctagBankFilename );
        ar & BOOST_SERIALIZATION_NVP( _cannyThrLow );
        ar & BOOST_SERIALIZATION_NVP( _cannyThrHigh );
        ar & BOOST_SERIALIZATION_NVP( _distSearch );
        ar & BOOST_SERIALIZATION_NVP( _angleVoting );
        ar & BOOST_SERIALIZATION_NVP( _ratioVoting );
        ar & BOOST_SERIALIZATION_NVP( _averageVoteMin);
        ar & BOOST_SERIALIZATION_NVP( _thrMedianDistanceEllipse);
        ar & BOOST_SERIALIZATION_NVP( _maximumNbSeeds);
        ar & BOOST_SERIALIZATION_NVP( _maximumNbCandidatesLoopTwo);
        ar & BOOST_SERIALIZATION_NVP( _numCrowns );
        ar & BOOST_SERIALIZATION_NVP( _minPointsSegmentCandidate );
        ar & BOOST_SERIALIZATION_NVP( _minVotesToSelectCandidate );
        ar & BOOST_SERIALIZATION_NVP( _threshRobustEstimationOfOuterEllipse );
        ar & BOOST_SERIALIZATION_NVP( _ellipseGrowingEllipticHullWidth );
        ar & BOOST_SERIALIZATION_NVP( _windowSizeOnInnerEllipticSegment );
        ar & BOOST_SERIALIZATION_NVP( _numberOfMultiresLayers );
        ar & BOOST_SERIALIZATION_NVP( _numberOfProcessedMultiresLayers );
        ar & BOOST_SERIALIZATION_NVP( _numCutsInIdentStep );
        ar & BOOST_SERIALIZATION_NVP( _numSamplesOuterEdgePointsRefinement );
        ar & BOOST_SERIALIZATION_NVP( _cutsSelectionTrials );
        ar & BOOST_SERIALIZATION_NVP( _sampleCutLength );
        ar & BOOST_SERIALIZATION_NVP( _minIdentProba );
        ar & BOOST_SERIALIZATION_NVP( _useLMDif );
        ar & BOOST_SERIALIZATION_NVP( _searchForAnotherSegment );
        ar & BOOST_SERIALIZATION_NVP( _writeOutput );
        _nCircles = 2*_numCrowns;
    }
};

}
}
}
}

#endif

