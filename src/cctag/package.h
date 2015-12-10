#pragma once

#include <iostream>
#include <string>

#include "cctag/fileDebug.hpp"
#include "cctag/visualDebug.hpp"
#include "cctag/progBase/exceptions.hpp"
#include "cctag/detection.hpp"
#include "cctag/view.hpp"
#include "cctag/image.hpp"
#include "cctag/types.hpp"
#include "cctag/EdgePoint.hpp"
// #include "cuda/ptrstep.h"
// #include "cctag/cmdline.hpp"

namespace popart
{
class FramePackage;

class Package
{
public:
    Package( const cctag::Parameters & params,
             const cctag::CCTagMarkersBank & bank );

    void lockPhase1( );
    void unlockPhase1( );
    void lockPhase2( ) { }
    void unlockPhase2( ) { }

    void resetTables( );

    void init( int frameId,
               const cv::Mat* src);

    FramePackage* getFramePackage( int i ) {
        if( i >= _numLayers ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                    << "    trying to access more than " << _numLayers << " allocations" << std::endl;
            exit( -1 );
        }
        if( not _framePackage ) {
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                    << "    FramePackages not allocated yet." << std::endl;
            exit( -1 );
        }
        return _framePackage[i];
    }

    void detection( std::ostream & output,
                    std::string debugFileName = "" );

private:
    int                            _frameId;
    const cv::Mat*                 _src;
    const cctag::Parameters&       _params;
    const cctag::CCTagMarkersBank& _bank;
    int                            _numLayers;
    FramePackage**                 _framePackage;

public:
    // temp storage for cctagMultiresDetection_inner
    cctag::WinnerMap*               winners;
    std::vector<cctag::EdgePoint*>* seeds;
};

} //namespace popart
