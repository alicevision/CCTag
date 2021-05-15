/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <boost/program_options.hpp>

#include <string>

namespace cctag {

class CmdLine
{
  public:
    std::string _filename{};
    std::string _cctagBankFilename{};
    std::string _paramsFilename{};
    std::size_t _nRings{3};
    std::string _outputFolderName{};
    bool _saveDetectedImage{false};
    bool _showUnreliableDetections{false};
#ifdef CCTAG_WITH_CUDA
    bool _switchSync{false};
    std::string _debugDir{};
    bool _useCuda{false};
    int _parallel{1};
#endif

    CmdLine();

    bool parse(int argc, char* argv[]);
    void print(const char* const argv0) const;

    void usage(const char* const argv0) const;

  private:
    boost::program_options::options_description _allParams;
};

}
