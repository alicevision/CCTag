#include "Regression.h"

static std::vector<boost::filesystem::path> CollectFiles(const boost::filesystem::path dirPath)
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

/////////////////////////////////////////////////////////////////////////////

TestRunner::TestRunner(const std::string& inputDir, const std::string& outputDir) :
  _inputDirPath(inputDir), _outputDirPath(outputDir)
{
  if (!exists(_inputDirPath) || !is_directory(_inputDirPath))
    throw std::runtime_error("TestRunner: inputDir is not a directory");
  if (!exists(_outputDirPath) || !is_directory(_outputDirPath))
    throw std::runtime_error("TestRunner: outputDir is not a directory");
  _inputFilePaths = CollectFiles(inputDir);
}

// Input directory must contain images.
void TestRunner::generateReferenceResults(const cctag::Parameters& parameters)
{
  size_t i = 1, count = _inputFilePaths.size();
  for (const auto& inputFilePath: _inputFilePaths) {
    std::clog << "Processing file " << i++ << "/" << count << ": " << inputFilePath << std::endl;
    FileLog fileLog = FileLog::detect(inputFilePath.native(), parameters);
    auto outputPath = _outputDirPath / inputFilePath.filename().replace_extension(".xml");
    fileLog.save(outputPath.native());
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
    fileLog.load(inputFilePath.native());
    fileLog = FileLog::detect(fileLog.filename, fileLog.parameters);
    auto outputPath = _outputDirPath / inputFilePath.filename();
    fileLog.save(outputPath.native());
  }
}

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
