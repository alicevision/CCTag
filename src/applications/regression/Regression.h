#pragma once

#include <boost/filesystem.hpp>
#include "TestLog.h"

class TestRunner
{
  const boost::filesystem::path _inputDirPath;
  const boost::filesystem::path _outputDirPath;
  std::vector<boost::filesystem::path> _inputFilePaths;
  
  void collectInputFiles();
  
public:
  TestRunner(const std::string& inputDir, const std::string& outputDir);
  void generateReferenceResults(const cctag::Parameters& parameters);
  void generateTestResults();
};
