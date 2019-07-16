/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <algorithm>
#include "Regression.h"

static void RemoveAllFiles(const boost::filesystem::path& dirPath);
static std::vector<boost::filesystem::path> CollectFiles(const boost::filesystem::path& dirPath);
static bool SortTags(FrameLog& log);

/////////////////////////////////////////////////////////////////////////////

TestRunner::TestRunner(const std::string& inputDir, const std::string& outputDir, boost::optional<bool> useCuda) :
  _inputDirPath(inputDir), _outputDirPath(outputDir), _useCuda(useCuda)
{
  if (!exists(_inputDirPath) || !is_directory(_inputDirPath))
    throw std::runtime_error("TestRunner: inputDir is not a directory");
  if (!exists(_outputDirPath) || !is_directory(_outputDirPath))
    throw std::runtime_error("TestRunner: outputDir is not a directory");
  RemoveAllFiles(_outputDirPath);
  _inputFilePaths = CollectFiles(_inputDirPath);
}

void TestRunner::adjustParameters(cctag::Parameters& parameters)
{
  if (_useCuda)
    parameters._useCuda = *_useCuda;
}

// Input directory must contain images.
// NB! parameters is by-val since we may need to adjust them.
void TestRunner::generateReferenceResults(cctag::Parameters parameters)
{
  adjustParameters(parameters);
  size_t i = 1, count = _inputFilePaths.size();
  for (const auto& inputFilePath: _inputFilePaths) {
    std::clog << "Processing file " << i++ << "/" << count << ": " << inputFilePath << std::endl;
    FileLog fileLog = FileLog::detect(inputFilePath.string(), parameters);
    const auto outputPath = _outputDirPath / inputFilePath.filename().replace_extension(".xml");
    fileLog.save(outputPath.string());
  }
}

// Input directory must contain XML files; parameters and input file will be read from those.
void TestRunner::generateTestResults()
{
  size_t i = 1, count = _inputFilePaths.size();
  for (const auto& inputFilePath: _inputFilePaths)
  if (inputFilePath.extension() == ".xml") {
    std::clog << "Processing file " << i++ << "/" << count << ": " << inputFilePath << std::endl;
    FileLog fileLog;
    fileLog.load(inputFilePath.string());
    adjustParameters(fileLog.parameters);
    fileLog = FileLog::detect(fileLog.filename, fileLog.parameters);
    auto outputPath = _outputDirPath / inputFilePath.filename();
    fileLog.save(outputPath.string());
  }
}

/////////////////////////////////////////////////////////////////////////////
// TestChecker assumption: all IDs in the frame are different.

TestChecker::TestChecker(const std::string& referenceDir, const std::string& testDir, float epsilon) :
  _referenceDirPath(referenceDir), _testDirPath(testDir), _epsilon(epsilon), _failed(false)
{
  if (!exists(_referenceDirPath) || !is_directory(_referenceDirPath))
    throw std::runtime_error("TestChecker: referenceDir is not a directory");
  if (!exists(_testDirPath) || !is_directory(_testDirPath))
    throw std::runtime_error("TestChecker: testDir is not a directory");
  _referenceFilePaths = CollectFiles(_referenceDirPath);
  _testFilePaths = CollectFiles(_testDirPath);
}

boost::filesystem::path TestChecker::testToReferencePath(const boost::filesystem::path& testPath)
{
  const auto testFilename = testPath.filename();
  auto it = std::find_if(_referenceFilePaths.begin(), _referenceFilePaths.end(),
    [testFilename](const boost::filesystem::path& p) { return p.filename() == testFilename; });
  return it != _referenceFilePaths.end() ? *it : boost::filesystem::path();
}

bool TestChecker::check()
{
  size_t i = 1, count = _testFilePaths.size();
  for (const auto& testFilePath: _testFilePaths) {
    std::clog << "Processing file " << i++ << "/" << count << ": " << testFilePath << std::endl;
    try {
      check(testFilePath);
    }
    catch (check_error& e) {
      std::clog << "  FAILED: " << e.what() << std::endl;
      _failed = true;
    }
  }
  return !_failed;
}

void TestChecker::check(const boost::filesystem::path& testFilePath)
{
  auto referenceFilePath = testToReferencePath(testFilePath);
  if (referenceFilePath.empty())
    throw check_error("reference file not found");

  FileLog referenceLog, testLog;
  referenceLog.load(referenceFilePath.string());
  testLog.load(testFilePath.string());
  compare(referenceLog, testLog);
}

void TestChecker::compare(FileLog& referenceLog, FileLog& testLog)
{
  const auto frameOrdCmp = [](const FrameLog& f1, const FrameLog& f2) { return f1.frame < f2.frame; };
  
  if (referenceLog.filename != testLog.filename)
    throw check_error("mismatching filenames");
  if (referenceLog.parameters._nCrowns != testLog.parameters._nCrowns)  // XXX: should check parameters better
    throw check_error("mismatching parameters");
  if (referenceLog.frameLogs.size() != testLog.frameLogs.size())
    throw check_error("mismatching frame counts");
  if (!std::is_sorted(referenceLog.frameLogs.begin(), referenceLog.frameLogs.end(), frameOrdCmp))
    throw check_error("reference log frames not monotonic");
  if (!std::is_sorted(testLog.frameLogs.begin(), testLog.frameLogs.end(), frameOrdCmp))
    throw check_error("test log frames not monotonic");
  
  const size_t frameCount = referenceLog.frameLogs.size();
  for (size_t i = 0; i < frameCount; ++i)
    compare(referenceLog.frameLogs[i], testLog.frameLogs[i], i);
}


void TestChecker::compare(FrameLog& referenceLog, FrameLog& testLog, size_t frame)
{
  if (referenceLog.tags.size() != testLog.tags.size())
    throw check_error(std::string("different # of tags in frame ") + std::to_string(frame));
  if (!SortTags(referenceLog))
    throw check_error("reference log contains duplicate IDs");
  if (!SortTags(testLog))
    throw check_error("test log contains duplicate IDs");
  
  _elapsedDiffAcc(testLog.elapsedTime - referenceLog.elapsedTime);
  
  const size_t tagCount = referenceLog.tags.size();
  for (size_t i = 0; i < tagCount; ++i) {
    _qualityDiffAcc(testLog.tags[i].quality - referenceLog.tags[i].quality);
    compare(referenceLog.tags[i], testLog.tags[i], frame);
  }
}

void TestChecker::compare(const DetectedTag& referenceTag, const DetectedTag& testTag, size_t frame)
{
  const bool ref_reliable = referenceTag.status == cctag::status::id_reliable;
  const bool test_reliable = testTag.status == cctag::status::id_reliable;
  
  if (ref_reliable ^ test_reliable)
    throw check_error(std::string("tags of different status in frame ") + std::to_string(frame));
  
  if (ref_reliable && test_reliable) {
    if (referenceTag.id != testTag.id)
      throw check_error(std::string("tags with different IDs frame ") + std::to_string(frame));

    float dx = fabs(referenceTag.x - testTag.x);
    float dy = fabs(referenceTag.y - testTag.y);
    if (dx > _epsilon || dy > _epsilon)
      throw check_error(std::string("tags at different positions in frame ") + std::to_string(frame));
  }
}

/////////////////////////////////////////////////////////////////////////////

static void RemoveAllFiles(const boost::filesystem::path& dirPath)
{
  using namespace boost::filesystem;
  remove_all(dirPath);
  create_directories(dirPath);
}

static std::vector<boost::filesystem::path> CollectFiles(const boost::filesystem::path& dirPath)
{
  using namespace boost::filesystem;
  std::vector<path> filePaths;
  
  directory_iterator it(dirPath), end;
  while (it != end) {
    auto de = *it++;
    if (de.status().type() == regular_file)
      filePaths.push_back(canonical(de.path()));
  }
  
  return filePaths;
}

// Removes all tags with status != 1, then sorts them by id. Returns true if there are no duplicate tags.
static bool SortTags(FrameLog& log)
{
  {
    auto it = std::remove_if(log.tags.begin(), log.tags.end(),
      [](const DetectedTag& t) { return t.status < 1; });
    log.tags.erase(it, log.tags.end());
  }
  
  std::sort(log.tags.begin(), log.tags.end(),
    [](const DetectedTag& t1, const DetectedTag& t2) { return t1.id < t2.id; });
  
  return std::adjacent_find(log.tags.begin(), log.tags.end(),
    [](const DetectedTag& t1, const DetectedTag& t2) { return t1.id == t2.id; }) == log.tags.end();
}
