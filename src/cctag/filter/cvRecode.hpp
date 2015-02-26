#ifndef _CCTAG_CVRECODE_HPP_
#define _CCTAG_CVRECODE_HPP_

#include <opencv2/core/types_c.h>

void cvRecodedCannyGPUFilter2D(
        void* srcarr,
        void* dstarr,
        CvMat*& dx,
        CvMat*& dy,
        double low_thresh,
        double high_thresh,
        int aperture_size);

void cvRecodedCanny(
        void* srcarr,
        void* dstarr,
        CvMat*& dx,
        CvMat*& dy,
        double low_thresh,
        double high_thresh,
        int aperture_size);

#endif

