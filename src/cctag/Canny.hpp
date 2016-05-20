#ifndef VISION_CCTAG_CANNY_HPP_
#define VISION_CCTAG_CANNY_HPP_

#include <cctag/Types.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <vector>


namespace cctag {

class EdgePoint;

void edgesPointsFromCanny(
        EdgePointCollection& edgeCollection,
        const cv::Mat & edges,
        const cv::Mat & dx,
        const cv::Mat & dy );

} // namespace cctag

#endif

