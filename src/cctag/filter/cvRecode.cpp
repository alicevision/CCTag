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
#include <algorithm>
#include <tuple>

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
  boost::timer t;  
  std::vector<uchar*> stack;
  uchar** stack_top = nullptr, ** stack_bottom = nullptr;
  
  const cv::Size size = imgGraySrc.size();
  int flags = aperture_size;
  int low, high;
  uchar* map;
  ptrdiff_t mapstep;
  int maxsize;
  int i, j;

  if( imgGraySrc.type() != CV_8UC1 ||
      imgCanny.type() != CV_8UC1 )
      CV_Error( CV_StsUnsupportedFormat, "" );

  if( !CV_ARE_SIZES_EQ( &imgGraySrc, &imgCanny ) )
       CV_Error( CV_StsUnmatchedSizes, "" );

  std::tie(low_thresh, high_thresh) = std::minmax(low_thresh, high_thresh);

  aperture_size &= INT_MAX;
  if( ( aperture_size & 1 ) == 0 || aperture_size < 3 || aperture_size > 7 )
    CV_Error( CV_StsBadFlag, "" );

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

  // defaults for cv::filter2D
  const cv::Point anchor{-1,-1};
  const double delta{0};

  {
    
    bool use1Dkernel = false;
    
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
      
      const cv::Mat kernelGau1D = (cv::Mat_<float>(9,1) << 
        0.000053390535453f,
        0.001768051711852f,
        0.021539279301849f,
        0.096532352630054f,
        0.159154943091895f,
        0.096532352630054f,
        0.021539279301849f,
        0.001768051711852f,
        0.000053390535453f);
      const cv::Mat kernelDGau1D = (cv::Mat_<float>(9,1) << 
        -0.002683701023220f,
        -0.066653979229454f,
        -0.541341132946452f,
        -1.213061319425269f,
        0.f,
        1.213061319425269f,
        0.541341132946452f,
        0.066653979229454f,
        0.002683701023220f);

      // Transpose guassian filter 
      const cv::Mat kernelGau1DT = kernelGau1D.t();
      
      // Transpose derivate of gaussian filter
      const cv::Mat kernelDGau1DT = kernelDGau1D.t();
      
      cv::Mat imgSmooth(imgDX.size(), CV_32FC1);
      cv::Mat imgTmp(imgDX.size(), CV_32FC1);
      cv::Mat dx_debug(imgDX.size(), CV_32FC1);
      cv::Mat dy_debug(imgDY.size(), CV_32FC1);
      
      CCTAG_COUT("before cvFilter2D 1");
      cv::filter2D(imgGraySrc, imgTmp, CV_32FC1, kernelGau1D, anchor, delta, cv::BORDER_REPLICATE);
      CCTAG_COUT("before cvFilter2D 2");
      cv::filter2D(imgTmp, imgSmooth, CV_32FC1, kernelGau1DT, anchor, delta, cv::BORDER_REPLICATE);
      
      CCTAG_COUT("before cvFilter2D 3");
      cv::filter2D(imgSmooth, imgTmp, CV_32FC1, kernelGau1D, anchor, delta, cv::BORDER_REPLICATE);
      
      CCTAG_COUT("before cvFilter2D 4");
      cv::filter2D(imgTmp, dx_debug, CV_32FC1, kernelDGau1DT, anchor, delta, cv::BORDER_REPLICATE);
      
      CCTAG_COUT("before cvFilter2D 5");
      cv::filter2D(imgSmooth, imgTmp, CV_32FC1, kernelGau1DT, anchor, delta, cv::BORDER_REPLICATE);
      CCTAG_COUT("before cvFilter2D 6");
      cv::filter2D(imgTmp, dy_debug, CV_32FC1, kernelDGau1D, anchor, delta, cv::BORDER_REPLICATE);  
      CCTAG_COUT("end");
      
      
      CCTAG_COUT("1D version : DX_DEBUG values");
      //CCTAG_COUT(dx_debug.rows);
      for (int i=0; i< dx_debug.rows ; ++i)
      {
        for (int j=0; j< dx_debug.cols ; ++j)
        {
          std::cout << std::fixed << std::setprecision(1) << dx_debug.ptr<float>(i)[j] << " ";
        }
        std::cout << std::endl;
      }
      CCTAG_COUT("1D version : END DX_DEBUG values");
      //cvTranspose( kerneldX, kerneldY );
      //cvFilter2D( src, dy, kernelGau1D );
      
    }else
    {
      // The second option is to apply the (9x9) 2D following kernel
      const cv::Mat kerneldX = -1.f * (cv::Mat_<float>(9,9) << 
        0.000000143284235f, 0.000003558691641f, 0.000028902492951f, 0.000064765993382f, 0.f, -0.000064765993382f, -0.000028902492951f, -0.000003558691641f, -0.000000143284235f,
        0.000004744922188f, 0.000117847682078f, 0.000957119116802f, 0.002144755142391f, 0.f, -0.002144755142391f, -0.000957119116802f, -0.000117847682078f, -0.000004744922188f,
        0.000057804985902f, 0.001435678675203f, 0.011660097860113f, 0.026128466569370f, 0.f, -0.026128466569370f, -0.011660097860113f, -0.001435678675203f, -0.000057804985902f,
        0.000259063973527f, 0.006434265427174f, 0.052256933138740f, 0.117099663048638f, 0.f, -0.117099663048638f, -0.052256933138740f, -0.006434265427174f, -0.000259063973527f,
        0.000427124283626f, 0.010608310271112f, 0.086157117207395f, 0.193064705260108f, 0.f, -0.193064705260108f, -0.086157117207395f, -0.010608310271112f, -0.000427124283626f,
        0.000259063973527f, 0.006434265427174f, 0.052256933138740f, 0.117099663048638f, 0.f, -0.117099663048638f, -0.052256933138740f, -0.006434265427174f, -0.000259063973527f,
        0.000057804985902f, 0.001435678675203f, 0.011660097860113f, 0.026128466569370f, 0.f, -0.026128466569370f, -0.011660097860113f, -0.001435678675203f, -0.000057804985902f,
        0.000004744922188f, 0.000117847682078f, 0.000957119116802f, 0.002144755142391f, 0.f, -0.002144755142391f, -0.000957119116802f, -0.000117847682078f, -0.000004744922188f,
        0.000000143284235f, 0.000003558691641f, 0.000028902492951f, 0.000064765993382f, 0.f, -0.000064765993382f, -0.000028902492951f, -0.000003558691641f, -0.000000143284235f);

      const cv::Mat kerneldY = kerneldX.t();

      cv::filter2D(imgGraySrc, imgDX, CV_16SC1, kerneldX, anchor, delta, cv::BORDER_REPLICATE);
      cv::filter2D(imgGraySrc, imgDY, CV_16SC1, kerneldY, anchor, delta, cv::BORDER_REPLICATE);    
    }
  }

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

#ifdef DEBUG_MAGMAP_BY_GRIFF
#ifdef USE_INTEGER_REP
  std::vector<int> mag_collect;
#else // USE_INTEGER_REP
  std::vector<float> mag_collect;
#endif // USE_INTEGER_REP
  std::ofstream* mag_img_file  = nullptr;
  std::ofstream* hyst_img_file = nullptr;

#ifdef WITH_CUDE
  if( params->_debugDir == "" ) {
    std::cerr << __FUNCTION__ << ":" << __LINE__
              << ": debugDir not set, not writing debug output" << std::endl;
  } else {
    std::cerr << __FUNCTION__ << ":" << __LINE__ << ": debugDir is ["
              << params->_debugDir << "] using that directory" << std::endl;

    std::ostringstream mag_img_name;
    mag_img_name << params->_debugDir << "cpu-" << debug_info_level << "-mag.pgm";
    mag_img_file = new std::ofstream( mag_img_name.str().c_str() );
    *mag_img_file << "P5" << std::endl
                  << size.width << " " << size.height << std::endl
                  << "255" << std::endl;

    std::ostringstream hyst_img_name;
    hyst_img_name << params->_debugDir << "cpu-" << debug_info_level << "-hyst.pgm";
    hyst_img_file = new std::ofstream( hyst_img_name.str().c_str() );
    *hyst_img_file << "P5" << std::endl
                   << size.width << " " << size.height << std::endl
                   << "255" << std::endl;
  }
#endif // WITH_CUDE
#endif // DEBUG_MAGMAP_BY_GRIFF

  cv::AutoBuffer<char> buffer;
  //               (             mag_buf                )   (                 map                  )
  buffer.allocate( ( size.width + 2 ) * 3 * sizeof( int ) + ( size.width + 2 ) * ( size.height + 2 ) );

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
    const short* _imgDX = imgDX.ptr<short>(i);
    const short* _imgDY = imgDY.ptr<short>(i);
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
          _mag[j] = abs( _imgDX[j] ) + abs( _imgDY[j] );
      } else {
        // Using Euclidian distance

        for( j = 0; j < size.width; j++ )
        {
          float* _magf = (float*)_mag;
          x = _imgDX[j];
          y = _imgDY[j];
#ifdef USE_INTEGER_REP
          _mag[j] = (int)rintf( (float)std::sqrt( (float)x * x + (float)y * y ) );
#else
          _magf[j] = (float)std::sqrt( (float)x * x + (float)y * y );
#endif
          CCTAG_IF_DEBUG(
            if (_imgDX[j] > 200){
              CCTAG_COUT_NOENDL(_imgDX[j] << ", " << _imgDY[j] << ", ");
            }
          )
        }
      }
    }
    else
      memset( _mag - 1, 0, ( size.width + 2 ) * sizeof( int ) );

#ifdef DEBUG_MAGMAP_BY_GRIFF
    if( mag_img_file ) {
      if( i > 0 ) {
        for( int j=0; j<size.width; j++ ) {
#ifdef USE_INTEGER_REP
            mag_collect.push_back( _mag[j] );
#else // USE_INTEGER_REP
            mag_collect.push_back( float(_mag[j]) );
#endif // USE_INTEGER_REP
        }
      }
    }
#endif // DEBUG_MAGMAP_BY_GRIFF

    // at the very beginning we do not have a complete ring
    // buffer of 3 magnitude rows for non-maxima suppression
    if( i == 0 )
      continue;

    _map     = map + mapstep * i + 1;
    _map[-1] = _map[size.width] = 1;

    _mag = mag_buf[1] + 1; // take the central row
    _imgDX = imgDX.ptr<short>(i - 1);
    _imgDY = imgDY.ptr<short>(i - 1);

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

      x = _imgDX[j];
      y = _imgDY[j];
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

#ifdef DEBUG_MAGMAP_BY_GRIFF
  if( mag_img_file ) {
#ifdef USE_INTEGER_REP
    std::vector<int>::iterator it;
    it = min_element( mag_collect.begin(), mag_collect.end() );
    int minval = *it;
    it = max_element( mag_collect.begin(), mag_collect.end() );
    int maxval = *it;
#else // USE_INTEGER_REP
    std::vector<float>::iterator it;
    it = min_element( mag_collect.begin(), mag_collect.end() );
    float minval = *it;
    it = max_element( mag_collect.begin(), mag_collect.end() );
    float maxval = *it;
#endif // USE_INTEGER_REP
    unsigned char* write_mag = new unsigned char[size.width * size.height];
    int idx=0;
    for( it = mag_collect.begin(); it!=mag_collect.end(); ++it ) {
      write_mag[idx++] = uint8_t( ( *it - minval ) * 256 / ( maxval - minval ) );
    }
    mag_img_file->write( (const char*)write_mag, size.width*size.height );
	delete[] write_mag;
  }
#endif // DEBUG_MAGMAP_BY_GRIFF
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
    uchar* _imgCanny = imgCanny.ptr<uchar>(i);

    for( j = 0; j < size.width; j++ )
    {
      _imgCanny[j] = ( uchar ) - ( _map[j] >> 1 );
    }
#ifdef DEBUG_MAGMAP_BY_GRIFF
    if( hyst_img_file )
        hyst_img_file->write( (const char*)_imgCanny, size.width );
#endif // DEBUG_MAGMAP_BY_GRIFF
  }

#ifdef DEBUG_MAGMAP_BY_GRIFF
  delete mag_img_file;
  delete hyst_img_file;
#endif // DEBUG_MAGMAP_BY_GRIFF
  DO_TALK( CCTAG_COUT_DEBUG( "Canny 4 : " << t.elapsed() ); )
}
