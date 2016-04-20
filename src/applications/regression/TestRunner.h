#pragma once

#include <boost/filesystem.hpp>
#include "TestLog.h"

class TestRunner
{
  const boost::filesystem::path _inputDirPath;
  const boost::filesystem::path _outputDirPath;
  const cctag::Parameters _parameters;
  std::vector<boost::filesystem::path> _inputFilePaths;
  
  void collectFiles();
  
public:
  TestRunner(const std::string& inputDir, const std::string& outputDir, const cctag::Parameters parameters);
  void detect();
};
