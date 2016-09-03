/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <getopt.h>
#include <iostream>
#include <string>
#include "CmdLine.hpp"

namespace cctag {

#define no_argument       0 
#define required_argument 1 
#define optional_argument 2


static const struct option longopts[] =
{
    {"input",      required_argument, 0, 'i'},
    {"nbrings",    required_argument, 0, 'n'},    
    {"bank",       required_argument, 0, 'b'},
    {"parameters", required_argument, 0, 'p'},
    {"output",     optional_argument, 0, 'o'},   
#ifdef WITH_CUDA
    {"sync",       no_argument,       0, 0xd0 },
    {"debug-dir",  required_argument, 0, 0xd1 },
    {"use-cuda",   no_argument,       0, 0xd2 },
    {"parallel",   required_argument, 0, 0xd3 },
#endif
    {0,0,0,0},
};

CmdLine::CmdLine( )
    : _filename( "" )
    , _nCrowns( "" )
    , _cctagBankFilename( "" )
    , _paramsFilename( "" )
    , _outputFolderName( "" )
#ifdef WITH_CUDA
    , _switchSync( false )
    , _debugDir( "" )
    , _useCuda( false )
    , _parallel( 1 )
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
    iarg = getopt_long(argc, argv, "i:n:b:p:o:", longopts, &index);

    switch (iarg)
    {
      case 'i'  : _filename          = optarg; has_i = true; break;
      case 'n'  : _nCrowns           = optarg; has_n = true; break;
      case 'b'  : _cctagBankFilename = optarg; break;
      case 'p'  : _paramsFilename    = optarg; break;
      case 'o'  : _outputFolderName  = optarg; break;
#ifdef WITH_CUDA
      case 0xd0 : _switchSync        = true;   break;
      case 0xd1 : _debugDir          = optarg; break;
      case 0xd2 : _useCuda           = true;   break;
      case 0xd3 : _parallel          = strtol( optarg, NULL, 0 );   break;
#endif
      default : break;
    }
  }
  return ( has_i & has_n );
}

void CmdLine::print( const char* const argv0 )
{
    std::cout << "You called " << argv0 << " with:" << std::endl
         << "    --input     " << _filename << std::endl
         << "    --nbrings     " << _nCrowns << std::endl
         << "    --bank      " << _cctagBankFilename << std::endl
         << "    --params    " << _paramsFilename << std::endl
         << "    --output    " << _outputFolderName << std::endl;
#ifdef WITH_CUDA
    if( _switchSync )
        std::cout << "    --sync " << std::endl;
    if( _debugDir != "" )
        std::cout << "    --debug-dir " << _debugDir << std::endl;
    if( _useCuda )
        std::cout << "    --use-cuda " << std::endl;
#endif
    std::cout << std::endl;
}

void CmdLine::usage( const char* const argv0 )
{
  std::cerr << "Usage: " << argv0 << "<parameters>\n"
          "    Mandatory:\n"
          "           [-i|--input] <imgpath>\n"
          "           [-n|--nbrings] <nbrings>\n"
          "    Optional:\n"
          "           [-p|--params <confpath>]\n"
          "           [-b|--bank] <bankpath>\n"
          "           [-o|--output] <outputfoldername>\n"
          "           [--sync]\n"
          "           [--debug-dir <debugdir>]\n"
          "           [--use-cuda]\n"
          "           [--parallel <n>]\n"
          "\n"
          "    <imgpath>  - path to an image (JPG, PNG) or video(avi, mov) or camera index for live capture (0, 1...)\n"
          "    <nbrings>  - number of rings of the CCTags to detect\n"
          "    <bankpath> - path to a bank parameter file, e.g. 4Crowns/ids.txt \n"
          "    <output>   - output folder name \n"
          "    <confpath> - path to configuration XML file \n"
          "    --sync     - CUDA debug option, run all CUDA ops synchronously\n"
          "    <debugdir> - path storing image to debug intermediate GPU results\n"
          "    --use-cuda - select GPU code instead of CPU code\n"
          "    --parallel - use <n> CUDA pipes concurrently (default 1)\n"
          "\n" << std::endl;
}

}

