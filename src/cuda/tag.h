#pragma once

#include <string>
#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/core.hpp>

#include "cuda/onoff.h"
#include "cctag/params.hpp"
#include "cctag/types.hpp"
#include "cctag/ImageCut.hpp"
#include "cctag/geometry/Ellipse.hpp"
#include "cctag/geometry/point.hpp"
#include "cctag/algebra/matrix/Matrix.hpp"
#include "cctag/logtime.hpp"

namespace popart
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h

class TagPipe
{
    std::vector<Frame*>  _frame;
public:
    void initialize( const uint32_t pix_w,
                     const uint32_t pix_h,
                     const cctag::Parameters& params,
                     cctag::logtime::Mgmt* durations );
    void load( unsigned char* pix );
    void tagframe( const cctag::Parameters& params );

    void convertToHost( size_t                          layer,
                        std::vector<cctag::EdgePoint>&  vPoints,
                        cctag::EdgePointsImage&         edgeImage,
                        std::vector<cctag::EdgePoint*>& seeds,
                        cctag::WinnerMap&               winners );

    inline std::size_t getNumOctaves( ) const {
        return _frame.size();
    }

    uint32_t getWidth(  size_t layer ) const;
    uint32_t getHeight( size_t layer ) const;

    cv::Mat* getPlane( size_t layer ) const;
    cv::Mat* getDx( size_t layer ) const;
    cv::Mat* getDy( size_t layer ) const;
    cv::Mat* getMag( size_t layer ) const;
    cv::Mat* getEdges( size_t layer ) const;

    double idCostFunction( int                                        level,
                           const cctag::numerical::geometry::Ellipse& ellipse,
                           const cctag::Point2dN<double>&             center,
                           std::vector<cctag::ImageCut>&              vCuts,
                           const size_t                               vCutMaxVecLen,
                           const float                                neighbourSize,
                           const size_t                               gridNSample,
                           cctag::Point2dN<double>&                   bestPoint,
                           cctag::numerical::BoundedMatrix3x3d&       bestHomographyOut );

    size_t getSignalBufferByteSize( int level ) const;

    void uploadCuts( int level, std::vector<cctag::ImageCut>& vCuts );

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

