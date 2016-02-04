#pragma once

#include <string>

namespace cctag {

class CmdLine
{
public:
    std::string _filename;
    std::string _cctagBankFilename;
    std::string _paramsFilename;
    std::string _nCrowns;
    std::string _outputFolderName;
#ifdef WITH_CUDA
    bool        _switchSync;
    std::string _debugDir;
    bool        _useCuda;
#endif

    CmdLine( );

    bool parse( int argc, char* argv[] );
    void print( const char* const argv0 );

    void usage( const char* const argv0 );
};

}

