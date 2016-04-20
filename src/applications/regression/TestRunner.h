#pragma once

#include <boost/filesystem.hpp>
#include "TestLog.h"

class TestRunner
{
  boost::filesystem::path _imageDir;
  boost::filesystem::path _resultDir;
  cctag::Parameters _parameters;
  
  std::vector<boost::filesystem::path> _imagePaths;
  
public:
  TestRunner(const std::string& imageDir, const std::string& resultDir, const cctag::Parameters parameters);
  void run();
};
