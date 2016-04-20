#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/program_options.hpp>

#include "Regression.h"

static std::string InputDir;
static std::string OtherDir;
static std::string ParametersFile;
static float Epsilon;

static std::string ParseOptions(int argc, char **argv)
{
  using namespace boost::program_options;
  std::string mode;
  
  options_description all_desc("Mode options");
  all_desc.add_options()
    ("generate-reference", "Generate reference results from a set of images")
    ("generate-test", "Generate test results based on reference results")
    ("check", "Check two sets of results")
    ("help", "Print help");
  
  options_description ref_desc("Generate reference options");
  ref_desc.add_options()
    ("input-dir", value<std::string>(&InputDir), "Input directory for images")
    ("output-dir", value<std::string>(&OtherDir), "Output directory for results [data will be overwritten!]")
    ("parameters", value<std::string>(&ParametersFile), "Detection parameters file");
  
  options_description test_desc("Generate test options");
  test_desc.add_options()
    ("reference-dir", value<std::string>(&InputDir), "Directory with reference results")
    ("output-dir", value<std::string>(&OtherDir), "Output directory for test results [data will be overwritten!]");
  
  options_description check_desc("Check options");
  check_desc.add_options()
    ("reference-dir", value<std::string>(&InputDir), "Directory with reference results")
    ("check-dir", value<std::string>(&OtherDir), "Directory with results to check")
    ("epsilon", value<float>(&Epsilon), "Position tolerance for x/y coordinates");
  
  all_desc.add(ref_desc).add(check_desc);
  
  variables_map vm;
  store(parse_command_line(argc, argv, all_desc), vm);
  
  if (vm.count("help")) {
    std::cout << all_desc << std::endl;
    exit(0);
  }
  
  if (vm.count("generate-reference")) {
    if (!vm.count("input-dir") || !vm.count("output-dir") || !vm.count("parameters"))
      throw error("All generate options are mandatory for --generate-reference");
    mode = "generate-reference";
  }
  
  if (vm.count("generate-test")) {
    if (!mode.empty())
      throw error("only one mode option can be specified");
    if (vm.count("parameters"))
      throw error("Cannot specify parameters for --generate-test");
    if (!vm.count("reference-dir") || !vm.count("output-dir"))
      throw error("All test options are mandatory for --generate-test");
    mode = "generate-test";
  }
  
  if (vm.count("check")) {
    if (!mode.empty())
      throw error("only one mode option can be specified");
    if (vm.count("parameters"))
      throw error("Cannot specify parameters for --generate-test");
    if (!vm.count("reference-dir") || !vm.count("check-dir") || !vm.count("epsilon"))
      throw error("All test options are mandatory for --check");
  }
  
  if (mode.empty())
    throw error("exactly one mode option must be specified");
  
  notify(vm);
  return mode;
}

static void GenerateReference()
{
  TestRunner testRunner(InputDir, OtherDir);
  cctag::Parameters parameters;
  
  {
    std::ifstream ifs(ParametersFile);
    boost::archive::xml_iarchive ia(ifs);
    ia >> boost::serialization::make_nvp("CCTagsParams", parameters);
  }

  testRunner.generateReferenceResults(parameters);
}

static bool ReportChecks()
{
  TestChecker testChecker(InputDir, OtherDir, Epsilon);
  bool ok = testChecker.check();
  
  if (ok) std::clog << "All checks PASSED" << std::endl;
  else std::clog << "Some checks FAILED" << std::endl;
  
  std::clog << "Performance difference report:\n";
  std::clog << "  time,    mean=" << testChecker.elapsedTimeDifferenceMean() << ",stdev=" << testChecker.elapsedTimeDifferenceStdev() << std::endl;
  std::clog << "  quality, mean=" << testChecker.qualityDifferenceMean() << ",stdev=" << testChecker.qualityDifferenceStdev() << std::endl;
}

int main(int argc, char **argv)
{
  try {
    auto mode = ParseOptions(argc, argv);
    
    if (mode == "generate-reference") {
      GenerateReference();
      return 0;
    }
    
    if (mode == "generate-test") {
      TestRunner testRunner(InputDir, OtherDir);
      testRunner.generateTestResults();
      return 0;
    }
    
    if (mode == "check") {
      bool ok = ReportChecks();
      return ok ? 0 : 1;
    }
    
    throw std::logic_error("internal error: invalid mode");
  }
  catch (boost::program_options::error& e) {
    std::cerr << "Failed to parse options: " << e.what() << std::endl;
    std::cerr << "Run with --help to see invocation synopsis." << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << "FATAL ERROR: " << e.what();
  }
  return 1;
}

