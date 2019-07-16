/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>

#include "frame.h"
#include "triple_point.h"

#include <random>


namespace cctag {

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
        using Byte_t = unsigned char;
        using map_t = std::map<Byte_t,RandomColor>;
        using it_t = map_t::iterator;
        using pair_t = std::pair<Byte_t,RandomColor>;

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
                               std::rand() % 255,
                               std::rand() % 255 );
                it = random_mapping.insert( pair_t( f, c ) ).first;
            }
            return it->second;
        }
    };

    static void writePGM( const std::string& filename,
                          const cv::cuda::PtrStepSzb& plane );

    template<class T>
    static void writePGMscaled_T( const std::string&            filename,
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
    static void writeASCII_T( const std::string&            filename,
                              const cv::cuda::PtrStepSz<T>& plane,
                              int                           width = 0 );
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
#ifndef NDEBUG
    static void writeASCII( const std::string&              filename,
                            const std::vector<TriplePoint>& list );
    static void writeASCII( const std::string& filename,
                            const std::string& info );
#endif // NDEBUG

    /* This may draw quite many lines. If there are too many, use skip to
     * draw only some.
     */
#ifndef NDEBUG
    static void plotLines( EdgeList<TriplePoint>& points,
                           int                    maxSize,
                           cv::cuda::PtrStepSzb   img,
                           bool                   normalize = true,
                           BaseColor              b = WHITE,
                           int                    skip = 0 );
#endif // NDEBUG
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

inline
DebugImage::BaseColor operator++( DebugImage::BaseColor& c )
{
    c = (DebugImage::BaseColor)( int(c)+1 % 256 );
    return c;
}

inline
DebugImage::BaseColor operator++( DebugImage::BaseColor& c, int )
{
    c = (DebugImage::BaseColor)( int(c)+1 % 256 );
    return c;
}

}; // namespace cctag

