/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/utils/Defines.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include "cctag/filter/cvRecode.hpp"
#include "cctag/Params.hpp"
#include "cctag/utils/Talk.hpp" // do DO_TALK macro

#include <boost/timer.hpp>

#include <cstdlib> // for ::abs
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#define DEBUG_MAGMAP_BY_GRIFF
#define USE_INTEGER_REP

void cvRecodedCanny(
    const cv::Mat & imgGraySrc,
    cv::Mat& imgCanny,
    cv::Mat& imgDX,
    cv::Mat& imgDY,
    float low_thresh,
    float high_thresh,
    int aperture_size,
    int debug_info_level,
    const cctag::Parameters* params )
{
    CvMat srcCvMat = imgGraySrc;
    CvMat *src = &srcCvMat;
  
    CvMat dstCvMat = imgCanny;
    CvMat *dst = &dstCvMat;
  
    CvMat dxCvMat = imgDX;
    CvMat *dx = &dxCvMat;
  
    CvMat dyCvMat = imgDY;
    CvMat *dy = &dyCvMat;
  
    boost::timer t;  
    std::vector<uchar*> stack;
    uchar** stack_top = nullptr, ** stack_bottom = nullptr;
  
    CvSize size;
    int flags = aperture_size;
    int low, high;
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    CvMat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ) )
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( low_thresh > high_thresh )
    {
        float t;
        CV_SWAP( low_thresh, high_thresh, t );
    }

    aperture_size &= INT_MAX;
    if( ( aperture_size & 1 ) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( CV_StsBadFlag, "" );

    size = CvSize(src->width,src->height);

    // TODO: no allocation here:
    //dx = cvCreateMat( size.height, size.width, CV_16SC1 );
    //dy = cvCreateMat( size.height, size.width, CV_16SC1 );
  
    // cvSobel is the function called by default in OpenCV, with a 3x3 kernel
    // to compute the derivative in x and y.
    // The kernel used to compute the derivative is changed here by a 9x9 one, to stick
    // with the results obtained with the canny implementation in the Matlab image
    // processing toolbox (2012)

    //	{
    //		dx = cvCreateMat( size.height, size.width, CV_32FC1 );
    //		dy = cvCreateMat( size.height, size.width, CV_32FC1 );
    //	}

    CCTAG_COUT_LILIAN( "Sobel allocation : " << t.elapsed() );
    t.restart();

    //cvSobel( src, dx, 1, 0, aperture_size );
    //cvSobel( src, dy, 0, 1, aperture_size );

#if 0
    // griff:
    // 1D kernel varation : this is broken - re-import the code from the working CUDA version
    if(use1Dkernel)
    {
      // ** Matlab code to generate the gaussian 1D kernel, filter size and standard deviation sigma**
      //    width = 4;
      //    sigma = 1;
      //    ssq = sigma^2;
      //    t = (-width:width);
      //    gaussian1D = exp(-(t.*t)/(2*ssq))/(2*pi*ssq)     % the gaussian 1D filter
      
//       float gaussian1D[9] = { 0.000053390535453, 
//                      0.001768051711852,
//                      0.021539279301849,
//                      0.096532352630054,
//                      0.159154943091895,
//                      0.096532352630054,
//                      0.021539279301849,
//                      0.001768051711852,
//                      0.000053390535453
//       };
       
      // ** Matlab code to generate the derivative of gaussian 1D kernel **
      //    t = -width:width;
      //    dgaussian1D = -t.*exp(-(t.*t)/(2*ssq))/(pi*ssq)/0.159154943091895 % the derivative of gaussian 1D filter
       
//      float dgaussian1D[9] = { 0.002683701023220,
//               0.066653979229454,
//               0.541341132946452,
//               1.213061319425269,
//               0,
//               -1.213061319425269,
//               -0.541341132946452,
//               -0.066653979229454,
//               -0.002683701023220 //1D gaussian derivative with sigma=1 divided by / 0.159154943091895
//       };
       
      // The first option is to apply successively the (above) 1D kernels (delivered the same result as the second option used below with a 9x9 2D kernel )
      // ** Matlab code on how to use the 1D kernels **   

      //    srcSmooth = imfilter(src, gaussian1D, 'conv','replicate');                  % srcSmooth = src X gaussian1D ; with dimensions(src)==dimensions(srcSmooth) (refered by 'replicate')
      //    srcSmooth = imfilter(srcSmooth, transpose(gaussian1D), 'conv','replicate'); % srcSmooth = srcSmooth x transpose(gaussian1D)
      //
      //    % Compute dx
      //    dx = imfilter(srcSmooth, transpose(gaussian1D), 'conv','replicate');        % dx = srcSmooth X transpose(gaussian1D)
      //    dx = imfilter(dx, dgaussian1D, 'conv','replicate');                         % dx = dx X dgaussian1D
      //
      //    % Compute dy
      //    dy = imfilter(srcSmooth, gaussian1D, 'conv','replicate');                   % dy = srcSmooth X gaussian1D
      //    dy = imfilter(dy, transpose(dgaussian1D), 'conv','replicate');              % dy = dy X transpose(dgaussian1D)
       
      // Summary of the two options
      // - First option using 1D kernels:
      //     1D convolution, kernel size 9 (6 times)

      // - Second option using 2D kernels:
      //     2D convolution, kernel size 9x9 (2 times)
      

      static const float kernelGau1D_in[9] = {
                            0.000053390535453f,
                            0.001768051711852f,
                            0.021539279301849f,
                            0.096532352630054f,
                            0.159154943091895f,
                            0.096532352630054f,
                            0.021539279301849f,
                            0.001768051711852f,
                            0.000053390535453f };
      
      static const float kernelDGau1D_in[9] = {
                            -0.002683701023220f,
                            -0.066653979229454f,
                            -0.541341132946452f,
                            -1.213061319425269f,
                             0.f,
                             1.213061319425269f,
                             0.541341132946452f,
                             0.066653979229454f,
                             0.002683701023220f };

      cv::Mat kernelGau1D ( 9, 1, CV_32FC1, (void*)kernelGau1D_in  );
      cv::Mat kernelDGau1D( 9, 1, CV_32FC1, (void*)kernelDGau1D_in );

      cv::Mat imgSmooth( imgDX.rows, imgDX.cols, CV_32FC1 );
      
      cv::Mat dx_debug( imgDX.rows, imgDX.cols, CV_32FC1 );
      cv::Mat dy_debug( imgDX.rows, imgDX.cols, CV_32FC1 );
      
      CCTAG_COUT("before cvSepFilter2D 1");
      cv::sepFilter2D( imgGraySrc, imgSmooth, CV_32FC1, kernelGau1D, kernelGau1D );
      
      CCTAG_COUT("before cvSepFilter2D 2");
      cv::sepFilter2D( imgSmooth, imgDX, CV_32FC1, kernelGau1D, kernelDGau1D );
      
      CCTAG_COUT("before cvSepFilter2D 3");
      cv::sepFilter2D( imgSmooth, imgDY, CV_32FC1, kernelGau1D, kernelDGau1D);
      CCTAG_COUT("end");
      
      
//       CCTAG_COUT("1D version : DX_DEBUG values");
      //CCTAG_COUT(dx_debug->rows);
//       for (int i=0; i< dx_debug->rows ; ++i)
//       {
//         for (int j=0; j< dx_debug->cols ; ++j)
//         {
//           std::cout << std::fixed << std::setprecision(1) << dx_debug->data.fl[ i*dx_debug->step + j] << " ";
//         }
//         std::cout << std::endl;
//       }
//       CCTAG_COUT("1D version : END DX_DEBUG values");
      //cvTranspose( kerneldX, kerneldY );
      //cvFilter2D( src, dy, kernelGau1D );
      
    }
#endif
    // The second option is to apply the (9x9) 2D following kernel

    CvMat* kerneldX = cvCreateMat( 9, 9, CV_32FC1 );
    CvMat* kerneldY = cvCreateMat( 9, 9, CV_32FC1 );

    CV_MAT_ELEM( *kerneldX, float, 0, 0 ) = 0.000000143284235f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 1 ) = 0.000003558691641f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 2 ) = 0.000028902492951f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 3 ) = 0.000064765993382f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 5 ) = -0.000064765993382f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 6 ) = -0.000028902492951f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 7 ) = -0.000003558691641f  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 8 ) = -0.000000143284235f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 0 ) = 0.000004744922188f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 1 ) = 0.000117847682078f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 2 ) = 0.000957119116802f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 3 ) = 0.002144755142391f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 5 ) = -0.002144755142391f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 6 ) = -0.000957119116802f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 7 ) = -0.000117847682078f  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 8 ) = -0.000004744922188f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 0 ) = 0.000057804985902f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 1 ) = 0.001435678675203f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 2 ) = 0.011660097860113f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 3 ) = 0.026128466569370f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 5 ) = -0.026128466569370f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 6 ) = -0.011660097860113f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 7 ) = -0.001435678675203f  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 8 ) = -0.000057804985902f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 0 ) = 0.000259063973527f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 1 ) = 0.006434265427174f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 2 ) = 0.052256933138740f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 3 ) = 0.117099663048638f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 5 ) = -0.117099663048638f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 6 ) = -0.052256933138740f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 7 ) = -0.006434265427174f  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 8 ) = -0.000259063973527f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 0 ) = 0.000427124283626f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 1 ) = 0.010608310271112f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 2 ) = 0.086157117207395f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 3 ) = 0.193064705260108f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 5 ) = -0.193064705260108f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 6 ) = -0.086157117207395f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 7 ) = -0.010608310271112f  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 8 ) = -0.000427124283626f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 0 ) = 0.000259063973527f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 1 ) = 0.006434265427174f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 2 ) = 0.052256933138740f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 3 ) = 0.117099663048638f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 5 ) = -0.117099663048638f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 6 ) = -0.052256933138740f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 7 ) = -0.006434265427174f  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 8 ) = -0.000259063973527f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 0 ) = 0.000057804985902f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 1 ) = 0.001435678675203f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 2 ) = 0.011660097860113f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 3 ) = 0.026128466569370f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 5 ) = -0.026128466569370f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 6 ) = -0.011660097860113f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 7 ) = -0.001435678675203f  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 8 ) = -0.000057804985902f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 0 ) = 0.000004744922188f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 1 ) = 0.000117847682078f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 2 ) = 0.000957119116802f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 3 ) = 0.002144755142391f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 5 ) = -0.002144755142391f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 6 ) = -0.000957119116802f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 7 ) = -0.000117847682078f  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 8 ) = -0.000004744922188f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 0 ) = 0.000000143284235f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 1 ) = 0.000003558691641f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 2 ) = 0.000028902492951f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 3 ) = 0.000064765993382f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 4 ) = 0.f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 5 ) = -0.000064765993382f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 6 ) = -0.000028902492951f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 7 ) = -0.000003558691641f  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 8 ) = -0.000000143284235f  ;

    cvConvertScale( kerneldX, kerneldX, -1.f );
    cvTranspose( kerneldX, kerneldY );

    cvFilter2D( src, dx, kerneldX );
    cvFilter2D( src, dy, kerneldY );

    //      CCTAG_COUT("DX_DEBUG values");
    //      CCTAG_COUT(dx_debug->rows);
    //      for (int i=0; i< dx->rows ; ++i)
    //      {
    //        for (int j=0; j< dx->cols ; ++j)
    //        {
    //          std::cout << dx->data.s[ i*dx->step + j] << " ";
    //        }
    //        std::cout << std::endl;
    //      }
    //      CCTAG_COUT("END DX_DEBUG values");
      
    cvReleaseMat( &kerneldX );
    cvReleaseMat( &kerneldY );

#ifndef USE_INTEGER_REP
    if( flags & CV_CANNY_L2_GRADIENT )
    {
        Cv32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low  = ul.i;
        high = uh.i;
    }
    else
#endif // USE_INTEGER_REP
    {
        low  = cvFloor( low_thresh );
        high = cvFloor( high_thresh );
    }

    cv::AutoBuffer<char> buffer;
    buffer.allocate( ( size.width + 2 ) * ( size.height + 2 ) + ( size.width + 2 ) * 3 * sizeof( int ) );

    int* mag_buf[3];
    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map        = (uchar*)( mag_buf[2] + size.width + 2 );
    mapstep    = size.width + 2;

    maxsize = MAX( 1 << 10, size.width * size.height / 10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag_buf[0], 0, ( size.width + 2 ) * sizeof( int ) );
    memset( map, 1, mapstep );
    memset( map + mapstep * ( size.height + 1 ), 1, mapstep );

#define CANNY_PUSH( d )    *( d ) = (uchar)2, *stack_top++ = ( d )
#define CANNY_POP( d )     ( d )  = *--stack_top

    mag_row = cvMat( 1, size.width, CV_32F );

    DO_TALK( CCTAG_COUT_DEBUG( "Canny 1 took: " << t.elapsed() ); );
    t.restart();

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag    = mag_buf[( i > 0 ) + 1] + 1;
        const short* _dx = (short*)( dx->data.ptr + dx->step * i );
        const short* _dy = (short*)( dy->data.ptr + dy->step * i );
        uchar* _map;
        int x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;

            if( !( flags & CV_CANNY_L2_GRADIENT ) ) {
                // Using Manhattan distance
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs( _dx[j] ) + abs( _dy[j] );
            } else {
                // Using Euclidian distance
                for( j = 0; j < size.width; j++ )
                {
                    float* _magf = (float*)_mag;
                    x = _dx[j];
                    y = _dy[j];
#ifdef USE_INTEGER_REP
                    _mag[j] = (int)rintf( (float)std::sqrt( (float)x * x + (float)y * y ) );
#else
                    _magf[j] = (float)std::sqrt( (float)x * x + (float)y * y );
#endif
                    if (_dx[j] > 200){
                        std::cout << _dx[j] << ", ";
                        std::cout << _dy[j] << ", ";
                    }
                }
            }
        }
        else
            memset( _mag - 1, 0, ( size.width + 2 ) * sizeof( int ) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map     = map + mapstep * i + 1;
        _map[-1] = _map[size.width] = 1;

        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)( dx->data.ptr + dx->step * ( i - 1 ) );
        _dy = (short*)( dy->data.ptr + dy->step * ( i - 1 ) );

        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( ( stack_top - stack_bottom ) + size.width > maxsize )
        {
            int sz = (int)( stack_top - stack_bottom );
            maxsize = MAX( maxsize * 3 / 2, maxsize + 8 );
            stack.resize( maxsize );
            stack_bottom = &stack[0];
            stack_top    = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
#define CANNY_SHIFT 15
#define TG22  (int)( 0.4142135623730950488016887242097 * ( 1 << CANNY_SHIFT ) + 0.5 )

            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];

            x = abs( x );
            y = abs( y );
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ( ( x + x ) << CANNY_SHIFT );

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j - 1] && m >= _mag[j + 1] )
                    {
                        if( m > high && !prev_flag && _map[j - mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j + magstep2] && m >= _mag[j + magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j - mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s] )
                    {
                        if( m > high && !prev_flag && _map[j - mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j]   = (uchar)1;
        }

        // scroll the ring buffer
        _mag       = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    DO_TALK( CCTAG_COUT_DEBUG( "Canny 2 took : " << t.elapsed() ); )

    t.restart();

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( ( stack_top - stack_bottom ) + 8 > maxsize )
        {
            int sz = (int)( stack_top - stack_bottom );
            maxsize = MAX( maxsize * 3 / 2, maxsize + 8 );
            stack.resize( maxsize );
            stack_bottom = &stack[0];
            stack_top    = stack_bottom + sz;
        }

        CANNY_POP( m );

        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep - 1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep + 1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep - 1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep + 1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    DO_TALK( CCTAG_COUT_DEBUG( "Canny 3 took : " << t.elapsed() ); )

    t.restart();

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep * ( i + 1 ) + 1;
        uchar* _dst       = dst->data.ptr + dst->step * i;

        for( j = 0; j < size.width; j++ )
        {
            _dst[j] = ( uchar ) - ( _map[j] >> 1 );
        }
    }

    DO_TALK( CCTAG_COUT_DEBUG( "Canny 4 : " << t.elapsed() ); )
}
