#pragma once

// #include <iostream>
// #include <string>

// #include "cctag/fileDebug.hpp"
// #include "cctag/visualDebug.hpp"
// #include "cctag/progBase/exceptions.hpp"
// #include "cctag/detection.hpp"
// #include "cctag/view.hpp"
// #include "cctag/image.hpp"
// #include "cctag/types.hpp"
// #include "cctag/EdgePoint.hpp"
#include "cuda/ptrstep.h"
// #include "cctag/cmdline.hpp"

namespace popart
{

class FramePackage
{
public:
    FramePackage( int width, int height );
    ~FramePackage( );

    void pinAll( );
    void unpinAll( );

private:
    int                     _w;
    int                     _h;

protected:
    cv::cuda::PtrStepSzb    _h_plane;
    cv::cuda::PtrStepSz16s  _h_dx;
    cv::cuda::PtrStepSz16s  _h_dy;
    cv::cuda::PtrStepSz32u  _h_mag;
    cv::cuda::PtrStepSzb    _h_edges;

    friend class Frame;
};

} //namespace popart
