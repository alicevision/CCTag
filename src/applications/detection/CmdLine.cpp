/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <string>
#include <memory>


#include "CmdLine.hpp"

namespace cctag {


CmdLine::CmdLine( ) :
        _allParams("Program for detecting CCTags in one image, in a directory or in a video\n"
               "For each image or video frame it detects the markers and it shows the image with a graphical overlay"
               "showing the center of the tag, its ID and the outer ellipse")
{
    using namespace boost::program_options;

    options_description required("Required input parameters");
    required.add_options()
        ("input,i", value<std::string>(&_filename)->required(), "Path to an image (JPG, PNG) or video (avi, mov) "
             "or camera index for live capture (0, 1...) or to a directory containing the images to process")
        ("nbrings,n", value<std::size_t>(&_nRings)->required(), "Number of rings of the CCTags to detect");

    options_description optional("Optional parameters");
    optional.add_options()
        ("bank,b", value<std::string>(&_cctagBankFilename)->default_value(_cctagBankFilename), "Path to a bank parameter file, e.g. 4Crowns/ids.txt")
        ("params,p", value<std::string>(&_paramsFilename)->default_value(_paramsFilename), "Path to configuration XML file")
        ("output,o", value<std::string>(&_outputFolderName)->default_value(_outputFolderName), "Output folder name")
        ("save-detected-image,s", bool_switch(&_saveDetectedImage), "Save an image with the graphical overlay of the detected tags. "
             "For single images the saved images will have a '_detected' suffix and it will be placed either in "
             "the current directory or in the directory given by --output. For videos a file named #####.png "
             "will be saved instead with the #s representing the zero-padded frame number, either in the current directory "
             "or in the directory given by --output.")
        ("show-unreliable,u", bool_switch(&_showUnreliableDetections), "Show the unreliable tags (marker id = -1)")
#ifdef CCTAG_WITH_CUDA
        ("sync", bool_switch(&_switchSync), "CUDA debug option, run all CUDA ops synchronously")
        ("use-cuda", bool_switch(&_useCuda), "Select GPU code instead of CPU code")
        ("debug-dir", value<std::string>(&_debugDir)->default_value(_debugDir), "Path storing image to debug intermediate GPU results")
        ("parallel", value<int>(&_parallel)->default_value(_parallel), "Use <n> CUDA pipes concurrently (default 1)")
#endif
            ;

    _allParams.add(required).add(optional);
}

bool CmdLine::parse(int argc, char* argv[])
{
    using namespace boost::program_options;

    variables_map vm;
    try
    {
        store(parse_command_line(argc, argv, _allParams), vm);
        if(vm.count("help") || (argc == 1))
        {
            return false;
        }
        notify(vm);
    }
    catch(const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}

void CmdLine::print(const char* const argv0) const
{
    std::cout << "You called " << argv0 << " with:" << std::endl
              << "    --input     " << _filename << std::endl
              << "    --nbrings     " << _nRings << std::endl
              << "    --bank      " << _cctagBankFilename << std::endl
              << "    --params    " << _paramsFilename << std::endl
              << "    --output    " << _outputFolderName << std::endl;
    if(_saveDetectedImage)
        std::cout << "    --save-detected-image" << std::endl;
    if(_showUnreliableDetections)
        std::cout << "    --show-unreliable" << std::endl;
#ifdef CCTAG_WITH_CUDA
    std::cout << "    --parallel " << _parallel << std::endl;
    if(_switchSync)
        std::cout << "    --sync " << std::endl;
    if(!_debugDir.empty())
        std::cout << "    --debug-dir " << _debugDir << std::endl;
    if(_useCuda)
        std::cout << "    --use-cuda " << std::endl;
#endif
    std::cout << std::endl;
}

void CmdLine::usage(const char* const argv0) const { std::cout << _allParams << std::endl; }

}
