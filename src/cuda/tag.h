#pragma once

#ifdef WITH_CUDA
#include <string>
#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/core.hpp>

#include "cctag/params.hpp"
#include "cctag/types.hpp"

namespace popart
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h

class TagPipe
{
    std::vector<Frame*>  _frame;
public:
    void initialize( const uint32_t pix_w,
                     const uint32_t pix_h,
                     const cctag::Parameters& params );
    void load( unsigned char* pix );
    void tagframe( const cctag::Parameters& params );
    void download( size_t                          layer,
                   std::vector<cctag::EdgePoint>&  vPoints,
                   cctag::EdgePointsImage&         edgeImage,
                   std::vector<cctag::EdgePoint*>& seeds,
                   cctag::WinnerMap&               winners );

    void debug( unsigned char* pix,
                const cctag::Parameters& params );

    static void debug_cpu_origin( int                      layer,
                                  const cv::Mat&           img,
                                  const cctag::Parameters& params );

    static void debug_cpu_edge_out( int                      layer,
                                    const cv::Mat&           edges,
                                    const cctag::Parameters& params );

    static void debug_cpu_dxdy_out( TagPipe*                 pipe,
                                    int                      layer,
                                    const cv::Mat&           cpu_dx,
                                    const cv::Mat&           cpu_dy,
                                    const cctag::Parameters& params );

    static void debug_cmp_edge_table( int                           layer,
                                      const cctag::EdgePointsImage& cpu,
                                      const cctag::EdgePointsImage& gpu,
                                      const cctag::Parameters&      params );
};

}; // namespace popart

#else // WITH_CUDA
namespace popart
{
typedef void* TagPipe;
};
#endif // WITH_CUDA
