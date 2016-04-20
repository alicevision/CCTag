#pragma once

#include <boost/filesystem.hpp>
#include "TestLog.h"

class TestRunner
{
  const boost::filesystem::path _inputDirPath;
  const boost::filesystem::path _outputDirPath;
  std::vector<boost::filesystem::path> _inputFilePaths;
  
public:
  TestRunner(const std::string& inputDir, const std::string& outputDir);
  void generateReferenceResults(const cctag::Parameters& parameters);
  void generateTestResults();
};

class TestChecker
{
  const boost::filesystem::path _referenceDirPath;
  const boost::filesystem::path _testDirPath;
  
public:
  TestChecker(const std::string& referenceDir, const std::string& testDir);
};