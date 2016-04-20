#include <iostream>
#include <string>
#include <boost/program_options.hpp>

static std::string InputDir;
static std::string OtherDir;
static std::string ParametersFile;

static std::string ParseOptions(int argc, char **argv)
{
  using namespace boost::program_options;
  std::string mode;
  
  options_description all_desc("Allowed options");
  all_desc.add_options()
    ("generate", "Generate results from a set of images")
    ("check", "Check two sets of results")
    ("help", "Print help");
  
  options_description gen_desc("Generate options");
  gen_desc.add_options()
    ("input-dir", value<std::string>(&InputDir), "Input directory for images")
    ("output-dir", value<std::string>(&OtherDir), "Output directory for results")
    ("parameters", value<std::string>(&ParametersFile), "Detection parameters file");
  
  options_description check_desc("Check options");
  check_desc.add_options()
    ("reference-dir", value<std::string>(&InputDir), "Directory with reference results")
    ("check-dir", value<std::string>(&OtherDir), "Directory with results to check");
  
  all_desc.add(gen_desc).add(check_desc);
  
  variables_map vm;
  store(parse_command_line(argc, argv, all_desc), vm);
  
  if (vm.count("help")) {
    std::cout << all_desc;
    exit(0);
  }
  
  if (vm.count("generate")) {
    if (!vm.count("input-dir") || !vm.count("output-dir") || !vm.count("parameters"))
      throw error("All generate options are mandatory for --generate");
    mode = "generate";
  }
  else if (vm.count("check")) {
    if (vm.count("parameters"))
      throw error("Cannot specify parameters for --check");
    if (!vm.count("reference-dir") || !vm.count("check-dir"))
      throw error("All check options are mandatory for --check");
    mode = "check";
  }
  else {
    throw error("generate or check mode must be specified");
  }
  
  notify(vm);
  return mode;
}

int main(int argc, char **argv)
{
  try {
    ParseOptions(argc, argv);
    return 0;
  }
  catch (boost::program_options::error& e) {
    std::cerr << "Failed to parse options: " << e.what() << std::endl;
    std::cerr << "Run with --help to see invocation synopsis." << std::endl;
  }
  return 1;
}

