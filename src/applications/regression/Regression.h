#pragma once

#include <stdexcept>
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
  const float _epsilon;
  
  using PathVector = std::vector<boost::filesystem::path>;
  
  struct check_error : public std::runtime_error
  {
    check_error(const std::string& what) : runtime_error(what)
    { }
  };

  PathVector _referenceFilePaths;
  PathVector _testFilePaths;
  bool _failed;
  
  void check(const boost::filesystem::path& testFilePath);
  void compare(FileLog& referenceLog, FileLog& testLog);
  void compare(FrameLog& referenceLog, FrameLog& testLog, size_t frame);
  boost::filesystem::path testToReferencePath(const boost::filesystem::path& testPath);
  
public:
  TestChecker(const std::string& referenceDir, const std::string& testDir, float epsilon);
  bool check();
};