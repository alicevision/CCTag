/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdexcept>
#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include "TestLog.h"

namespace bacc = boost::accumulators;

class TestRunner
{
  const boost::filesystem::path _inputDirPath;
  const boost::filesystem::path _outputDirPath;
  const boost::optional<bool> _useCuda;
  std::vector<boost::filesystem::path> _inputFilePaths;
  
  void adjustParameters(cctag::Parameters& parameters);
  
public:
  TestRunner(const std::string& inputDir, const std::string& outputDir, boost::optional<bool> useCuda);
  void generateReferenceResults(cctag::Parameters parameters);
  void generateTestResults();
};

class TestChecker
{
  const boost::filesystem::path _referenceDirPath;
  const boost::filesystem::path _testDirPath;
  const float _epsilon;
  
  using PathVector = std::vector<boost::filesystem::path>;
  
  struct check_error : public std::runtime_error
  {
    explicit check_error(const std::string& what) : runtime_error(what)
    { }
  };

  PathVector _referenceFilePaths;
  PathVector _testFilePaths;
  bacc::accumulator_set<float,
    bacc::stats<bacc::tag::mean,
                bacc::tag::variance>> _elapsedDiffAcc;  // over all frames in the dataset
  bacc::accumulator_set<float,
    bacc::stats<bacc::tag::mean,
                bacc::tag::variance>> _qualityDiffAcc;  // over all tags in the dataset
  bool _failed;
  
  void check(const boost::filesystem::path& testFilePath);
  void compare(FileLog& referenceLog, FileLog& testLog);
  void compare(FrameLog& referenceLog, FrameLog& testLog, size_t frame);
  void compare(const DetectedTag& referenceTag, const DetectedTag& testTag, size_t frame);
  boost::filesystem::path testToReferencePath(const boost::filesystem::path& testPath);
  
public:
  TestChecker(const std::string& referenceDir, const std::string& testDir, float epsilon);
  bool check();
  float elapsedTimeDifferenceMean() { return bacc::mean(_elapsedDiffAcc); }
  float elapsedTimeDifferenceStdev() { return sqrt(bacc::variance(_elapsedDiffAcc)); }
  float qualityDifferenceMean() { return bacc::mean(_qualityDiffAcc); }
  float qualityDifferenceStdev() { return sqrt(bacc::variance(_qualityDiffAcc)); }
};
