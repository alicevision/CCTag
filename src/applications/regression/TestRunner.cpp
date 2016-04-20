#include "TestRunner.h"

TestRunner::TestRunner(const std::string& inputDir, const std::string& outputDir, const cctag::Parameters parameters) :
  _inputDirPath(inputDir), _outputDirPath(outputDir), _parameters(parameters)
{
  collectFiles();
}

void TestRunner::collectFiles()
{
  using namespace boost::filesystem;

  if (!exists(_inputDirPath) || !is_directory(_inputDirPath))
    throw std::runtime_error("TestRunner: inputDir is not a directory");
  if (!exists(_outputDirPath) || !is_directory(_outputDirPath))
    throw std::runtime_error("TestRunner: outputDir is not a directory");
  
  directory_iterator it(_inputDirPath), end;
  
  while (it != end) {
    auto de = *it++;
    if (de.status().type() == regular_file)
      _inputFilePaths.push_back(canonical(de.path()));
  }
}

void TestRunner::detect()
{
  size_t i = 1, count = _inputFilePaths.size();
  for (const auto& inputFilePath: _inputFilePaths) {
    std::clog << "Processing file " << i++ << "/" << count << ": " << inputFilePath;
    FileLog fileLog = FileLog::detect(inputFilePath.native(), _parameters);
    auto outputPath = _outputDirPath / inputFilePath.filename().replace_extension(".xml");
    fileLog.save(outputPath.native());
  }
}
