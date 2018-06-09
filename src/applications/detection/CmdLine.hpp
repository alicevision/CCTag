/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <string>
#include <boost/program_options.hpp>

namespace cctag {

class CmdLine
{
public:
    std::string _filename;
    std::string _cctagBankFilename;
    std::string _paramsFilename;
    std::size_t _nRings;
    std::string _outputFolderName;
#ifdef WITH_CUDA
    bool        _switchSync;
    std::string _debugDir;
    bool        _useCuda;
    int         _parallel;
#endif

    CmdLine( );

    bool parse( int argc, char* argv[] );
    void print( const char* const argv0 );

    void usage( const char* const argv0 );

private:
	boost::program_options::options_description _allParams;
};

}

