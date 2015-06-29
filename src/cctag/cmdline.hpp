#pragma once

#include <string>

class CmdLine
{
public:
    std::string filename;
    std::string cctagBankFilename;
    std::string paramsFilename;
#ifdef WITH_CUDA
    bool        switchSync;
    std::string debugDir;
#endif

    CmdLine( );

    bool parse( int argc, char* argv[] );
    void print( const char* const argv0 );

    void usage( const char* const argv0 );
};

extern CmdLine cmdline;

