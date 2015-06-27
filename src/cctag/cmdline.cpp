#include <getopt.h>
#include <iostream>
#include <string>
#include "cmdline.hpp"

using namespace std;

#define no_argument       0 
#define required_argument 1 
#define optional_argument 2

CmdLine cmdline;

static const struct option longopts[] =
{
    {"input",      required_argument, 0, 'i'},
    {"bank",       required_argument, 0, 'b'},
    {"parameters", required_argument, 0, 'p'},
    {"sync",       no_argument,       0, 0xd0 },
    {"debug-dir",  required_argument, 0, 0xd1 },
    {0,0,0,0},
};

CmdLine::CmdLine( )
    : filename( "" )
    , cctagBankFilename( "" )
    , paramsFilename( "" )
#ifdef WITH_CUDA
    , switchSync( false )
    , debugDir( "" )
#endif
{ }

bool CmdLine::parse( int argc, char* argv[] )
{
  int index;
  int iarg=0;

  // bools to check that mandatory parameters are present
  bool has_i = false;
  bool has_b = false;

  //turn off getopt error message
  // opterr=1; 

  while(iarg != -1)
  {
    iarg = getopt_long(argc, argv, "i:b:p:", longopts, &index);

    switch (iarg)
    {
      case 'i'  : filename          = optarg; has_i = true; break;
      case 'b'  : cctagBankFilename = optarg; has_b = true; break;
      case 'p'  : paramsFilename    = optarg; break;
#ifdef WITH_CUDA
      case 0xd0 : switchSync        = true;   break;
      case 0xd1 : debugDir          = optarg; break;
#endif
      default : break;
    }
  }
  return ( has_i & has_b );
}

void CmdLine::print( const char* const argv0 )
{
    cout << "You called " << argv0 << " with:" << endl
         << "    --input     " << filename << endl
         << "    --bank      " << cctagBankFilename << endl
         << "    --params    " << paramsFilename << endl;
#ifdef WITH_CUDA
    if( switchSync )
        cout << "    --sync " << endl;
    if( debugDir != "" )
        cout << "    --debug-dir " << debugDir << endl;
#endif
    cout << endl;
}

void CmdLine::usage( const char* const argv0 )
{
  cerr << "Usage: " << argv0 << "<parameters>\n"
          "    Mandatory:\n"
          "           (-i|--input) <imgpath>\n"
          "           (-b|--bank) <bankpath>\n"
          "    Optional:\n"
          "           [-p|--params <confpath>]\n"
          "           [--sync]\n"
          "           [--debug-dir <debugdir>]\n"
          "\n"
          "    <imgpath>  - path to an image (JPG, PNG) or video\n"
          "    <bankpath> - path to a bank parameter file, e.g. 4Crowns/ids.txt \n"
          "    <confpath> - path to configuration XML file \n"
          "    --sync     - CUDA debug option, run all CUDA ops synchronously\n"
          "    <debugdir> - path storing image to debug intermediate results\n"
          "\n" << endl;
}

