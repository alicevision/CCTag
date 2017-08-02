/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/program_options.hpp>
#include "Regression.h"

static std::string SourceDir;
static std::string DestinationDir;
static std::string ParametersFile;
static float Epsilon;
static boost::optional<bool> UseCuda;

static std::string ParseOptions(int argc, char **argv)
{
  using namespace boost::program_options;
  std::string mode;
  
  options_description all_desc("Mode options");
  all_desc.add_options()
    ("gen-ref", "Generate reference results from images in the source directory")
    ("gen-test", "Generate test results based on settings from reference results in the source directory")
    ("compare", "Compare reference results in the source directory with results in the destination directory")
    ("use-cuda", value<bool>()->notifier([](bool v) { UseCuda = v; }),
      "Overrides implementation specified by parameters")
    ("help", "Print help");
  
  options_description data_desc("Data specification options");
  data_desc.add_options()
    ("src-dir", value<std::string>(&SourceDir), "Source directory")
    ("dst-dir", value<std::string>(&DestinationDir), "Destination directory [WARNING: current contents will be lost]")
    ("parameters", value<std::string>(&ParametersFile), "Detection parameters file")
    ("epsilon", value<float>(&Epsilon)->default_value(0.5f), "Position tolerance for x/y coordinates");
  
  all_desc.add(data_desc);
  
  variables_map vm;
  store(parse_command_line(argc, argv, all_desc), vm);
  
  if (vm.count("help")) {
    std::cout << all_desc << std::endl;
    std::cout << "WARNING: 'generate' modes overwrite data in the destination directory!" << std::endl;
    std::cout << std::endl;
    exit(0);
  }
  
  if (vm.count("gen-ref")) {
    if (!vm.count("src-dir") || !vm.count("dst-dir") || !vm.count("parameters"))
      throw error("generate-reference: input-dir, output-dir and parameters are mandatory");
    mode = "gen-ref";
  }
  
  if (vm.count("gen-test")) {
    if (!mode.empty())
      throw error("only one mode option can be specified");
    if (vm.count("parameters"))
      throw error("Cannot specify parameters for gen-test");
    if (!vm.count("src-dir") || !vm.count("dst-dir"))
      throw error("gen-test: src-dir and dst-dir are mandatory");
    mode = "gen-test";
  }
  
  if (vm.count("compare")) {
    if (!mode.empty())
      throw error("only one mode option can be specified");
    if (vm.count("parameters"))
      throw error("Cannot specify parameters for compare");
    if (!vm.count("src-dir") || !vm.count("dst-dir"))
      throw error("check: src-dir and dst-dir are mandatory");
    mode = "compare";
  }
  
  if (mode.empty())
    throw error("exactly one mode option must be specified");
  
  notify(vm);
  return mode;
}

static void GenerateReference()
{
  TestRunner testRunner(SourceDir, DestinationDir, UseCuda);
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
  TestChecker testChecker(SourceDir, DestinationDir, Epsilon);
  bool ok = testChecker.check();
  
  if (ok) std::clog << "All checks PASSED" << std::endl;
  else std::clog << "Some checks FAILED" << std::endl;
  
  std::clog << "Performance difference report:\n";
  std::clog << "  time,    mean=" << testChecker.elapsedTimeDifferenceMean() << ",stdev=" << testChecker.elapsedTimeDifferenceStdev() << std::endl;
  std::clog << "  quality, mean=" << testChecker.qualityDifferenceMean() << ",stdev=" << testChecker.qualityDifferenceStdev() << std::endl;
  
  return ok;
}

int main(int argc, char **argv)
{
  try {
    auto mode = ParseOptions(argc, argv);
    
    if (UseCuda) std::clog << "CUDA override SET to: " << *UseCuda << std::endl;
    else std::clog << "CUDA override NOT SET; will use parameters" << std::endl;
    
    if (mode == "gen-ref") {
      GenerateReference();
      return EXIT_SUCCESS;
    }
    
    if (mode == "gen-test") {
      TestRunner testRunner(SourceDir, DestinationDir, UseCuda);
      testRunner.generateTestResults();
      return EXIT_SUCCESS;
    }
    
    if (mode == "compare") {
      bool ok = ReportChecks();
      return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    throw std::logic_error("internal error: invalid mode");
  }
  catch (boost::program_options::error& e) {
    std::cerr << "Failed to parse options: " << e.what() << std::endl;
    std::cerr << "Run with --help to see invocation synopsis." << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << "FATAL ERROR: " << e.what() << std::endl;
  }
  return EXIT_FAILURE;
}

