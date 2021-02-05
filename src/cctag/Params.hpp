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
#include <sys/stat.h>  // needed for stat and mkdir
#include <sys/types.h> // needed for stat and mkdir

#include <cmath>
#include <cstddef>
#include <string>

#define NO_WEIGHT 0
#define INV_GRAD_WEIGHT 1
#define INV_SQRT_GRAD_WEIGHT 2
#define INV_SQUARE_GRAD_WEIGHT 3

namespace cctag {

static constexpr std::size_t kDefaultDistSearch = 30;
static constexpr std::size_t kDefaultNCrowns = 3;  // 4;
static constexpr std::size_t kDefaultNCircles = 6; // 8;
static constexpr int kDefaultThrGradientMagInVote = 2500;
static constexpr float kDefaultAngleVoting = 0.0f;
static constexpr float kDefaultRatioVoting = 4.f;
static constexpr float kDefaultAverageVoteMin = 0.f;
static constexpr float kDefaultThrMedianDistanceEllipse = 3.0f;
static constexpr std::size_t kDefaultMaximumNbSeeds = 500;
static constexpr std::size_t kDefaultMaximumNbCandidatesLoopTwo = 40; // 30;
static constexpr float kDefaultCannyThrLow = 0.01f;                   // 0.002
static constexpr float kDefaultCannyThrHigh = 0.04f;                  // 0.04
static constexpr std::size_t kDefaultMinPointsSegmentCandidate = 10;
static constexpr std::size_t kDefaultMinVotesToSelectCandidate = 3;
static constexpr float kDefaultThreshRobustEstimationOfOuterEllipse = 30.0f;
static constexpr float kDefaultEllipseGrowingEllipticHullWidth = 2.3f;
static constexpr std::size_t kDefaultWindowSizeOnInnerEllipticSegment = 20;
static constexpr std::size_t kDefaultNumberOfMultiresLayers = 4;
static constexpr std::size_t kDefaultNumberOfProcessedMultiresLayers = 4;
static constexpr std::size_t kDefaultNSamplesOuterEllipse = 150;
static constexpr std::size_t kDefaultNumCutsInIdentStep = 22; // 30;//100;//15;
static constexpr std::size_t kDefaultNumSamplesOuterEdgePointsRefinement = 20;
static constexpr std::size_t kDefaultCutsSelectionTrials = 500; // 10000;
static constexpr std::size_t kDefaultSampleCutLength = 100;
///  must be odd otherwise the ellipse center will not be included  in the nearby points.
static constexpr std::size_t kDefaultImagedCenterNGridSample = 5;
static constexpr float kDefaultImagedCenterNeighbourSize = 0.20f;
static constexpr float kDefaultMinIdentProba = 1e-6f; // 1e-6
static constexpr bool kDefaultUseLMDif = true;
static constexpr bool kDefaultSearchForAnotherSegment = true;
static constexpr bool kDefaultWriteOutput = false;
static constexpr bool kDefaultDoIdentification = true;
static constexpr uint32_t kDefaultMaxEdges = 20000;
#ifdef CCTAG_WITH_CUDA
static constexpr bool kDefaultUseCuda = true;
#else
static constexpr bool kDefaultUseCuda = false;
#endif
static constexpr size_t kDefaultPinnedCounters     = 100;
static constexpr size_t kDefaultPinnedNearbyPoints = 60;

static const std::string kParamCannyThrLow("kParamCannyThrLow");
static const std::string kParamCannyThrHigh("kParamCannyThrHigh");
static const std::string kParamDistSearch("kParamDistSearch");
static const std::string kThrGradientMagInVote("kThrGradientMagInVote");
static const std::string kParamAngleVoting("kParamAngleVoting");
static const std::string kParamRatioVoting("kParamRatioVoting");
static const std::string kParamAverageVoteMin("kParamAverageVoteMin");
static const std::string kParamThrMedianDistanceEllipse("kParamThrMedianDistanceEllipse");
static const std::string kParamMaximumNbSeeds("kParamMaximumNbSeeds");
static const std::string kParamMaximumNbCandidatesLoopTwo("kParamMaximumNbCandidatesLoopTwo");
static const std::string kParamNCrowns("kParamNCrowns");
static const std::string kParamMinPointsSegmentCandidate("kParamMinPointsSegmentCandidate");
static const std::string kParamMinVotesToSelectCandidate("kParamMinVotesToSelectCandidate");
static const std::string kParamThreshRobustEstimationOfOuterEllipse("kParamThreshRobustEstimationOfOuterEllipse");
static const std::string kParamEllipseGrowingEllipticHullWidth("kParamEllipseGrowingEllipticHullWidth");
static const std::string kParamWindowSizeOnInnerEllipticSegment("kParamWindowSizeOnInnerEllipticSegment");
static const std::string kParamNumberOfMultiresLayers("kParamNumberOfMultiresLayers");
static const std::string kParamNumberOfProcessedMultiresLayers("kParamNumberOfProcessedMultiresLayers");
static const std::string kParamNSamplesOuterEllipse("kParamNSamplesOuterEllipse");
static const std::string kParamNumCutsInIdentStep("kParamNumCutsInIdentStep");
static const std::string kParamNumSamplesOuterEdgePointsRefinement("kParamNumSamplesOuterEdgePointsRefinement");
static const std::string kParamCutsSelectionTrials("kParamCutsSelectionTrials");
static const std::string kParamSampleCutLength("kParamSampleCutLength");
static const std::string kParamImagedCenterNGridSample("kParamImagedCenterNGridSample");
static const std::string kParamImagedCenterNeighbourSize("kParamImagedCenterNeighbourSize");
static const std::string kParamMinIdentProba("kParamMinIdentProba");
static const std::string kParamUseLMDif("kParamUseLMDif");
static const std::string kParamSearchForAnotherSegment("kParamSearchForAnotherSegment");
static const std::string kParamWriteOutput("kParamWriteOutput");
static const std::string kParamDoIdentification("kParamDoIdentification");
static const std::string kParamMaxEdges("kParamMaxEdges");
static const std::string kUseCuda("kUseCuda");
static const std::string kPinnedCounters("kPinnedCounters");
static const std::string kPinnedNearbyPoints("kPinnedNearbyPoints");

static const std::size_t kWeight = INV_GRAD_WEIGHT;

/**
 * @brief Structure containing all the major parameters using in the CCTag detection algorithms.
 */
struct Parameters
{
    friend class boost::serialization::access;

    static bool OverrideChecked;
    static bool OverrideLoaded;
    static Parameters Override;
    static void LoadOverride();

    /**
     * @brief The constructor, normally the most interesting parameter is the number of crowns.
     * @param nCrowns The number of crowns that the markers to detect are made up of.
     */
    explicit Parameters(std::size_t nCrowns = kDefaultNCrowns);

    ///  canny low threshold
    float _cannyThrLow;
    ///  canny high threshold
    float _cannyThrHigh;
    ///  maximum distance (in pixels) of research from one edge points
    /// to another one. maximum length of a arc segment composing the polygonal line.
    std::size_t _distSearch;
    ///  during the voting procedure, the gradient direction is followed as long as the gradient magnitude is greater
    ///  than this threshold
    int _thrGradientMagInVote;
    ///  maximum angle between of gradient directions of two consecutive edge points.
    float _angleVoting;
    ///  maximum distance ratio between of gradient directions of two consecutive edge points
    float _ratioVoting;
    float _averageVoteMin;
    float _thrMedianDistanceEllipse;
    ///  number of seeds to process as potential candidates
    std::size_t _maximumNbSeeds;
    std::size_t _maximumNbCandidatesLoopTwo;
    ///  number of crowns
    std::size_t _nCrowns;
    ///  number of circles that can be detected based on \p _nCrowns, i.e. _nCircles = 2 * _nCrowns
    std::size_t _nCircles;
    ///  minimal number of points on the outer ellipse to select an inner segment candidate
    std::size_t _minPointsSegmentCandidate;
    ///  minimum number of received votes to select an edge point as a new seed.
    std::size_t _minVotesToSelectCandidate;
    ///  LMeDs threshold on robust estimation of the outer ellipse
    float _threshRobustEstimationOfOuterEllipse;
    ///  width of elliptic hull in ellipse growing
    float _ellipseGrowingEllipticHullWidth;
    ///  window size on the inner elliptic segment
    std::size_t _windowSizeOnInnerEllipticSegment;
    ///  number of multi-resolution layers
    std::size_t _numberOfMultiresLayers;
    ///  number of processed layers in multi-resolution
    std::size_t _numberOfProcessedMultiresLayers;
    ///  Number of points considered in the preliminary sampling in identification
    std::size_t _nSamplesOuterEllipse;
    ///  number of cuts in the identification step
    std::size_t _numCutsInIdentStep;
    ///  number of sample for the outer edge points refinement in identification
    std::size_t _numSamplesOuterEdgePointsRefinement;
    ///  number of trials in cuts selection
    std::size_t _cutsSelectionTrials;
    ///  sample cut length
    std::size_t _sampleCutLength;
    ///  number of point sample along the square grid side (e.g. 5 for a grid of 5x5=25 points)
    std::size_t _imagedCenterNGridSample;
    ///  grid width relatively to the length of longest semi-axis of the outer ellipse
    float _imagedCenterNeighbourSize;
    ///  minimal probability of delivered by the identification algorithm to consider a candidate as a CCTag
    float _minIdentProba;
    bool _useLMDif;
    ///  is CCTag can be made of many flow components (2). This implies an assembling step which may be time time
    ///  consuming.
    bool _searchForAnotherSegment;
    bool _writeOutput;
    ///  perform the identification step
    bool _doIdentification;
    ///  max number of edge point, determines memory allocation
    uint32_t _maxEdges;
    ///  if compiled with CCTAG_WITH_CUDA, whether to use cuda algorithm or not, otherwise it is ignored
    bool _useCuda;
    /// if _useCuda, physical memory reserved for internal counters, otherwise unused
    size_t _pinnedCounters;
    /// if _useCuda, physical memory reserved for point detection, otherwise unused
    size_t _pinnedNearbyPoints;
    ///  prefix for debug output
    std::string _debugDir;

    /**
     * @brief Serialize the parameter settings.
     * @tparam Archive The class to use to store the data.
     * @param[in,out] ar The object where to store the data.
     * @param[in] version The serialization version. 
     */
    template<class Archive>
    void serialize(Archive& ar, unsigned int version)
    {
        ar& BOOST_SERIALIZATION_NVP(_cannyThrLow);
        ar& BOOST_SERIALIZATION_NVP(_cannyThrHigh);
        ar& BOOST_SERIALIZATION_NVP(_distSearch);
        ar& BOOST_SERIALIZATION_NVP(_thrGradientMagInVote);
        ar& BOOST_SERIALIZATION_NVP(_angleVoting);
        ar& BOOST_SERIALIZATION_NVP(_ratioVoting);
        ar& BOOST_SERIALIZATION_NVP(_averageVoteMin);
        ar& BOOST_SERIALIZATION_NVP(_thrMedianDistanceEllipse);
        ar& BOOST_SERIALIZATION_NVP(_maximumNbSeeds);
        ar& BOOST_SERIALIZATION_NVP(_maximumNbCandidatesLoopTwo);
        ar& BOOST_SERIALIZATION_NVP(_nCrowns);
        ar& BOOST_SERIALIZATION_NVP(_minPointsSegmentCandidate);
        ar& BOOST_SERIALIZATION_NVP(_minVotesToSelectCandidate);
        ar& BOOST_SERIALIZATION_NVP(_threshRobustEstimationOfOuterEllipse);
        ar& BOOST_SERIALIZATION_NVP(_ellipseGrowingEllipticHullWidth);
        ar& BOOST_SERIALIZATION_NVP(_windowSizeOnInnerEllipticSegment);
        ar& BOOST_SERIALIZATION_NVP(_numberOfMultiresLayers);
        ar& BOOST_SERIALIZATION_NVP(_numberOfProcessedMultiresLayers);
        ar& BOOST_SERIALIZATION_NVP(_nSamplesOuterEllipse);
        ar& BOOST_SERIALIZATION_NVP(_numCutsInIdentStep);
        ar& BOOST_SERIALIZATION_NVP(_numSamplesOuterEdgePointsRefinement);
        ar& BOOST_SERIALIZATION_NVP(_cutsSelectionTrials);
        ar& BOOST_SERIALIZATION_NVP(_sampleCutLength);
        ar& BOOST_SERIALIZATION_NVP(_imagedCenterNGridSample);
        ar& BOOST_SERIALIZATION_NVP(_imagedCenterNeighbourSize);
        ar& BOOST_SERIALIZATION_NVP(_minIdentProba);
        ar& BOOST_SERIALIZATION_NVP(_useLMDif);
        ar& BOOST_SERIALIZATION_NVP(_searchForAnotherSegment);
        ar& BOOST_SERIALIZATION_NVP(_writeOutput);
        ar& BOOST_SERIALIZATION_NVP(_doIdentification);
        ar& BOOST_SERIALIZATION_NVP(_maxEdges);
        ar& BOOST_SERIALIZATION_NVP(_useCuda);
        ar& BOOST_SERIALIZATION_NVP(_pinnedCounters);
        ar& BOOST_SERIALIZATION_NVP(_pinnedNearbyPoints);
        _nCircles = 2 * _nCrowns;
    }

    /**
     * @brief Set the debug directory where debug data is stored
     * @param[in] debugDir
     */
    void setDebugDir(const std::string& debugDir);

    /**
     * @brief Whether to use the Cuda implementation or not.
     * @param[in] val \p true to use the Cuda implementation, \p false to use the CPU.
     * @note Ignored if the code is not built with Cuda support.
     */
    void setUseCuda(bool val);
};

} // namespace cctag
