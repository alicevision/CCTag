#pragma once

#include "cuda/onoff.h"

#include <string>
#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>

#include "cuda/tag_threads.h"
#include "cctag/Params.hpp"
#include "cctag/Types.hpp"
#include "cctag/ImageCut.hpp"
#include "cctag/geometry/Ellipse.hpp"
#include "cctag/geometry/Point.hpp"

namespace cctag { namespace logtime { struct Mgmt; } };

namespace popart
{

class Frame; // forward decl means cctag/*.cpp need not recompile for frame.h
class NearbyPoint;

class TagPipe
{
    std::vector<Frame*>         _frame;
    const cctag::Parameters&    _params;
    TagThreads                  _threads;
    std::vector<cudaStream_t>   _tag_streams;

public:
    TagPipe( const cctag::Parameters& params );

    void initialize( const uint32_t pix_w,
                     const uint32_t pix_h,
                     cctag::logtime::Mgmt* durations );
    void release( );
    void load( unsigned char* pix );
    void tagframe( );
    void handleframe( int layer );

    void convertToHost( size_t                          layer,
                        cctag::EdgePointCollection&     edgeCollection,
                        std::vector<cctag::EdgePoint*>& seeds);

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

    void checkTagAllocations( const int                numTags,
                              const cctag::Parameters& params );

    void imageCenterOptLoop(
        const int                                  tagIndex,
        const cctag::numerical::geometry::Ellipse& ellipse,
        const cctag::Point2d<Eigen::Vector3f>&             center,
        const int                                  vCutSize,
        const cctag::Parameters&                   params,
        NearbyPoint*                               cctag_pointer_buffer );

    bool imageCenterRetrieve(
        const int                                  tagIndex,
        cctag::Point2d<Eigen::Vector3f>&                   center,
        Eigen::Matrix3f&       bestHomographyOut,
        const cctag::Parameters&                   params,
        NearbyPoint*                               cctag_pointer_buffer );

    // size_t getSignalBufferByteSize( int level ) const;

    void uploadCuts( int                                 numTags,
                     const std::vector<cctag::ImageCut>* vCuts,
                     const cctag::Parameters&            params );

    void makeCudaStreams( int numTags );

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

