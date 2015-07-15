#pragma once

// #include <cuda_runtime.h>
// #include <assert.h>
// #include <string>
// #include <vector>

// #include <opencv2/core/cuda_types.hpp>

#include "frame.h"
// #include "../cctag/params.hpp"
// #include "frame_vote.h"
#include "triple_point.h"

namespace popart {

/* The class DebugImage could be a namespace, but I like the private
 * section.
 */
class DebugImage
{
public:
    enum BaseColor
    {
        BLACK = 0,
        GREY1,
        GREY2,
        GREY3,
        GREEN,
        BLUE,
        GREEN_1,
        GREEN_2,
        GREEN_3,
        GREEN_4,
        LAST,
        WHITE = 255
    };

    struct RandomColor
    {
        unsigned char r;
        unsigned char g;
        unsigned char b;

        RandomColor( unsigned char r_, unsigned char g_, unsigned char b_ )
            : r(r_), g(g_), b(b_)
            { }
    };

    struct RandomColorMap
    {
        typedef unsigned char                 Byte_t;
        typedef std::map<Byte_t,RandomColor>  map_t;
        typedef map_t::iterator               it_t;
        typedef std::pair<Byte_t,RandomColor> pair_t;

        map_t random_mapping;

        RandomColorMap( )
        {
            random_mapping.insert( pair_t( BLACK,   RandomColor(  0,  0,  0) ) );
            random_mapping.insert( pair_t( GREY1,   RandomColor(100,100,100) ) );
            random_mapping.insert( pair_t( GREY2,   RandomColor(150,150,150) ) );
            random_mapping.insert( pair_t( GREY3,   RandomColor(200,200,200) ) );
            random_mapping.insert( pair_t( GREEN,   RandomColor(  0,255,  0) ) );
            random_mapping.insert( pair_t( BLUE,    RandomColor(  0,  0,255) ) );
            random_mapping.insert( pair_t( GREEN_1, RandomColor(  0,255, 50) ) );
            random_mapping.insert( pair_t( GREEN_2, RandomColor(  0,255,100) ) );
            random_mapping.insert( pair_t( GREEN_3, RandomColor(  0,255,150) ) );
            random_mapping.insert( pair_t( GREEN_4, RandomColor(  0,255,200) ) );
        }

        const RandomColor& get( unsigned char f )
        {
            it_t it = random_mapping.find(f);
            if( it == random_mapping.end() )
            {
                RandomColor c( 255,
                               random() % 255,
                               random() % 255 );
                it = random_mapping.insert( pair_t( f, c ) ).first;
            }
            return it->second;
        }
    };

    static void writePGM( const std::string& filename,
                          const cv::cuda::PtrStepSzb& plane );

    template<class T>
    static void writePGMscaled_T( const std::string& filename,
                                  const cv::cuda::PtrStepSz<T>& plane );
    static void writePGMscaled( const std::string& filename,
                                const cv::cuda::PtrStepSz<float>& plane );
    static void writePGMscaled( const std::string& filename,
                                const cv::cuda::PtrStepSz<uint8_t>& plane );
    static void writePGMscaled( const std::string& filename,
                                const cv::cuda::PtrStepSz<int16_t>& plane );
    static void writePGMscaled( const std::string& filename,
                                const cv::cuda::PtrStepSz<uint32_t>& plane );

    static void writePPM( const std::string& filename,
                          const cv::cuda::PtrStepSzb& plane );

    template<class T>
    static void writeASCII_T( const std::string& filename,
                              const cv::cuda::PtrStepSz<T>& plane );
    static void writeASCII( const std::string& filename,
                            const cv::cuda::PtrStepSz<float>& plane );
    static void writeASCII( const std::string& filename,
                            const cv::cuda::PtrStepSz<uint8_t>& plane );
    static void writeASCII( const std::string& filename,
                            const cv::cuda::PtrStepSz<int16_t>& plane );
    static void writeASCII( const std::string& filename,
                            const cv::cuda::PtrStepSz<uint32_t>& plane );
    static void writeASCII( const std::string&      filename,
                            const std::vector<int>& list );
    static void writeASCII( const std::string&       filename,
                            const std::vector<int2>& list );
    static void writeASCII( const std::string&              filename,
                            const std::vector<TriplePoint>& list );

    /* This may draw quite many lines. If there are too many, use skip to
     * draw only some.
     */
    static void plotLines( EdgeList<TriplePoint>& points,
                           int                    maxSize,
                           cv::cuda::PtrStepSzb   img,
                           bool                   normalize = true,
                           BaseColor              b = WHITE,
                           int                    skip = 0 );
    static void plotPoints( const std::vector<TriplePoint>& v,
                            cv::cuda::PtrStepSzb            img,
                            bool                            normalize = true,
                            enum BaseColor                  b = WHITE );
    static void plotPoints( const std::vector<int2>& v,
                            cv::cuda::PtrStepSzb     img,
                            bool                     normalize = true,
                            enum BaseColor           b = WHITE );

public:
    /* static global variable
     */
    static RandomColorMap randomColorMap;

private:
    /* if normalize is true, set all non-zero (non-BLACK) pixels
     * in the image to 1 (GREY1).
     * else do nothing
     */
    static void normalizeImage( cv::cuda::PtrStepSzb img,
                                bool                 normalize );

    static int getColor( BaseColor b );

    static void plotOneLine( int2 from, int2 to, cv::cuda::PtrStepSzb img, int color );
};

}; // namespace popart

