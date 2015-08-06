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
    {"nbrings",    required_argument, 0, 'n'},    
    {"bank",       required_argument, 0, 'b'},
    {"parameters", required_argument, 0, 'p'},
#ifdef WITH_CUDA
    {"sync",       no_argument,       0, 0xd0 },
    {"debug-dir",  required_argument, 0, 0xd1 },
    {"use-cuda",   no_argument,       0, 0xd2 },
#endif
    {0,0,0,0},
};

CmdLine::CmdLine( )
    : _filename( "" )
    , _nCrowns( "" )
    , _cctagBankFilename( "" )
    , _paramsFilename( "" )
#ifdef WITH_CUDA
    , _switchSync( false )
    , _debugDir( "" )
    , _useCuda( false )
#endif
{ }

bool CmdLine::parse( int argc, char* argv[] )
{
  int index;
  int iarg=0;

  // bools to check that mandatory parameters are present
  bool has_i = false;
  bool has_n = false;

  //turn off getopt error message
  // opterr=1; 

  while(iarg != -1)
  {
    iarg = getopt_long(argc, argv, "i:n:b:p:", longopts, &index);

    switch (iarg)
    {
      case 'i'  : _filename          = optarg; has_i = true; break;
      case 'n'  : _nCrowns           = optarg; has_n = true; break;
      case 'b'  : _cctagBankFilename = optarg; break;
      case 'p'  : _paramsFilename    = optarg; break;
#ifdef WITH_CUDA
      case 0xd0 : _switchSync        = true;   break;
      case 0xd1 : _debugDir          = optarg; break;
      case 0xd2 : _useCuda           = true;   break;
#endif
      default : break;
    }
  }
  return ( has_i & has_n );
}

void CmdLine::print( const char* const argv0 )
{
    cout << "You called " << argv0 << " with:" << endl
         << "    --input     " << _filename << endl
         << "    --nbrings     " << _nCrowns << endl
         << "    --bank      " << _cctagBankFilename << endl
         << "    --params    " << _paramsFilename << endl;
#ifdef WITH_CUDA
    if( _switchSync )
        cout << "    --sync " << endl;
    if( _debugDir != "" )
        cout << "    --debug-dir " << _debugDir << endl;
    if( _useCuda )
        cout << "    --use-cuda " << endl;
#endif
    cout << endl;
}

void CmdLine::usage( const char* const argv0 )
{
  cerr << "Usage: " << argv0 << "<parameters>\n"
          "    Mandatory:\n"
          "           [-i|--input] <imgpath>\n"
          "           [-n|--nbrings] <nbrings>\n"
          "    Optional:\n"
          "           [-p|--params <confpath>]\n"
          "           [-b|--bank] <bankpath>\n"
          "           [--sync]\n"
          "           [--debug-dir <debugdir>]\n"
          "           [--use-cuda]\n"
          "\n"
          "    <imgpath>  - path to an image (JPG, PNG) or video\n"
          "    <nbrings>  - number of rings of the CCTags to detect\n"
          "    <bankpath> - path to a bank parameter file, e.g. 4Crowns/ids.txt \n"
          "    <confpath> - path to configuration XML file \n"
          "    --sync     - CUDA debug option, run all CUDA ops synchronously\n"
          "    <debugdir> - path storing image to debug intermediate results\n"
          "    --use-cuda - select GPU code instead of CPU code\n"
          "\n" << endl;
}

