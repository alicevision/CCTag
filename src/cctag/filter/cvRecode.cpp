#include "cvRecode.hpp"

#include <cctag/global.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/internal.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>

#include <boost/gil/image_view.hpp>
#include <boost/timer.hpp>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>


#ifdef with_cuda
void cvRecodedCannyGPUFilter2D( void* srcarr, void* dstarr, CvMat*& dx, CvMat*& dy,
                                double low_thresh, double high_thresh,
                                int aperture_size )
{
  boost::timer t;

  cv::AutoBuffer<char> buffer;
  std::vector<uchar*> stack;
  uchar** stack_top = 0, ** stack_bottom = 0;

  CvMat srcstub, * src = cvGetMat( srcarr, &srcstub );
  CvMat dststub, * dst = cvGetMat( dstarr, &dststub );
  CvSize size;
  int flags = aperture_size;
  int low, high;
  int* mag_buf[3];
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
    double t;
    CV_SWAP( low_thresh, high_thresh, t );
  }

  aperture_size &= INT_MAX;
  if( ( aperture_size & 1 ) == 0 || aperture_size < 3 || aperture_size > 7 )
    CV_Error( CV_StsBadFlag, "" );

  size = cvGetMatSize( src );

  {
    dx = cvCreateMat( size.height, size.width, CV_32FC1 );
    dy = cvCreateMat( size.height, size.width, CV_32FC1 );
  }

  ROM_COUT_LILIAN( "Sobel allocation : " << t.elapsed() );

  t.restart();

  {
    using namespace boost::gil;
    using namespace boost::numeric;
    ublas::matrix<double, ublas::column_major> kerneldY( 9, 9 );
    ublas::matrix<double, ublas::column_major> kerneldX( 9, 9 );
    ublas::matrix<int, ublas::column_major> kerneldYi( 9, 9 );
    ublas::matrix<int, ublas::column_major> kerneldXi( 9, 9 );

    kerneldY( 0, 0 ) = 0.000000143284235  ;
    kerneldY( 0, 1 ) = 0.000003558691641  ;
    kerneldY( 0, 2 ) = 0.000028902492951  ;
    kerneldY( 0, 3 ) = 0.000064765993382  ;
    kerneldY( 0, 4 ) = 0  ;
    kerneldY( 0, 5 ) = -0.000064765993382  ;
    kerneldY( 0, 6 ) = -0.000028902492951  ;
    kerneldY( 0, 7 ) = -0.000003558691641  ;
    kerneldY( 0, 8 ) = -0.000000143284235;
    kerneldY( 1, 0 ) = 0.000004744922188  ;
    kerneldY( 1, 1 ) = 0.000117847682078  ;
    kerneldY( 1, 2 ) = 0.000957119116802  ;
    kerneldY( 1, 3 ) = 0.002144755142391  ;
    kerneldY( 1, 4 ) = 0  ;
    kerneldY( 1, 5 ) = -0.002144755142391  ;
    kerneldY( 1, 6 ) = -0.000957119116802  ;
    kerneldY( 1, 7 ) = -0.000117847682078  ;
    kerneldY( 1, 8 ) = -0.000004744922188;
    kerneldY( 2, 0 ) = 0.000057804985902  ;
    kerneldY( 2, 1 ) = 0.001435678675203  ;
    kerneldY( 2, 2 ) = 0.011660097860113  ;
    kerneldY( 2, 3 ) = 0.026128466569370  ;
    kerneldY( 2, 4 ) = 0  ;
    kerneldY( 2, 5 ) = -0.026128466569370  ;
    kerneldY( 2, 6 ) = -0.011660097860113  ;
    kerneldY( 2, 7 ) = -0.001435678675203  ;
    kerneldY( 2, 8 ) = -0.000057804985902;
    kerneldY( 3, 0 ) = 0.000259063973527  ;
    kerneldY( 3, 1 ) = 0.006434265427174  ;
    kerneldY( 3, 2 ) = 0.052256933138740  ;
    kerneldY( 3, 3 ) = 0.117099663048638  ;
    kerneldY( 3, 4 ) = 0  ;
    kerneldY( 3, 5 ) = -0.117099663048638  ;
    kerneldY( 3, 6 ) = -0.052256933138740  ;
    kerneldY( 3, 7 ) = -0.006434265427174  ;
    kerneldY( 3, 8 ) = -0.000259063973527;
    kerneldY( 4, 0 ) = 0.000427124283626  ;
    kerneldY( 4, 1 ) = 0.010608310271112  ;
    kerneldY( 4, 2 ) = 0.086157117207395  ;
    kerneldY( 4, 3 ) = 0.193064705260108  ;
    kerneldY( 4, 4 ) = 0  ;
    kerneldY( 4, 5 ) = -0.193064705260108  ;
    kerneldY( 4, 6 ) = -0.086157117207395  ;
    kerneldY( 4, 7 ) = -0.010608310271112  ;
    kerneldY( 4, 8 ) = -0.000427124283626;
    kerneldY( 5, 0 ) = 0.000259063973527  ;
    kerneldY( 5, 1 ) = 0.006434265427174  ;
    kerneldY( 5, 2 ) = 0.052256933138740  ;
    kerneldY( 5, 3 ) = 0.117099663048638  ;
    kerneldY( 5, 4 ) = 0  ;
    kerneldY( 5, 5 ) = -0.117099663048638  ;
    kerneldY( 5, 6 ) = -0.052256933138740  ;
    kerneldY( 5, 7 ) = -0.006434265427174  ;
    kerneldY( 5, 8 ) = -0.000259063973527;
    kerneldY( 6, 0 ) = 0.000057804985902  ;
    kerneldY( 6, 1 ) = 0.001435678675203  ;
    kerneldY( 6, 2 ) = 0.011660097860113  ;
    kerneldY( 6, 3 ) = 0.026128466569370  ;
    kerneldY( 6, 4 ) = 0  ;
    kerneldY( 6, 5 ) = -0.026128466569370  ;
    kerneldY( 6, 6 ) = -0.011660097860113  ;
    kerneldY( 6, 7 ) = -0.001435678675203  ;
    kerneldY( 6, 8 ) = -0.000057804985902;
    kerneldY( 7, 0 ) = 0.000004744922188  ;
    kerneldY( 7, 1 ) = 0.000117847682078  ;
    kerneldY( 7, 2 ) = 0.000957119116802  ;
    kerneldY( 7, 3 ) = 0.002144755142391  ;
    kerneldY( 7, 4 ) = 0  ;
    kerneldY( 7, 5 ) = -0.002144755142391  ;
    kerneldY( 7, 6 ) = -0.000957119116802  ;
    kerneldY( 7, 7 ) = -0.000117847682078  ;
    kerneldY( 7, 8 ) = -0.000004744922188;
    kerneldY( 8, 0 ) = 0.000000143284235  ;
    kerneldY( 8, 1 ) = 0.000003558691641  ;
    kerneldY( 8, 2 ) = 0.000028902492951  ;
    kerneldY( 8, 3 ) = 0.000064765993382  ;
    kerneldY( 8, 4 ) = 0  ;
    kerneldY( 8, 5 ) = -0.000064765993382  ;
    kerneldY( 8, 6 ) = -0.000028902492951  ;
    kerneldY( 8, 7 ) = -0.000003558691641  ;
    kerneldY( 8, 8 ) = -0.000000143284235;

    kerneldX = ublas::trans( kerneldY );
    const int divisor = 100000;
    kerneldYi = kerneldY * divisor;
    kerneldXi = kerneldX * divisor;

    gray8_view_t svw = interleaved_view( src->cols, src->rows, (gray8_pixel_t*)src->data.fl, src->cols * sizeof(bits8) );
    gray32f_view_t vdx = interleaved_view( dx->cols, dx->rows, (gray32f_pixel_t*)dx->data.fl, dx->cols * sizeof(bits32f) );
    gray32f_view_t vdy = interleaved_view( dy->cols, dy->rows, (gray32f_pixel_t*)dy->data.fl, dy->cols * sizeof(bits32f) );

    popart::graphics::cuda::sobel( svw, vdx, vdy, kerneldXi, kerneldYi, divisor );
  }
  //cvShowImage("Sobel", dx);
  //cvWaitKey(0);

  ROM_COUT_LILIAN( "Sobel took: " << t.elapsed() );

  t.restart();

  if( flags & CV_CANNY_L2_GRADIENT )
  {
    Cv32suf ul, uh;
    ul.f = (float)low_thresh;
    uh.f = (float)high_thresh;

    low  = ul.i;
    high = uh.i;
  }
  else
  {
    low  = cvFloor( low_thresh );
    high = cvFloor( high_thresh );
  }

  buffer.allocate( ( size.width + 2 ) * ( size.height + 2 ) + ( size.width + 2 ) * 3 * sizeof( int ) );

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

  /* sector numbers
     (Top-Left Origin)

      1   2   3
   *  *  *
   * * *
      0*******0
   * * *
   *  *  *
      3   2   1
   */

#define CANNY_PUSH( d )    *( d ) = (uchar)2, *stack_top++ = ( d )
#define CANNY_POP( d )     ( d )  = *--stack_top

  mag_row = cvMat( 1, size.width, CV_32F );

  ROM_COUT_LILIAN( "Canny 1 took: " << t.elapsed() );

  t.restart();

  // calculate magnitude and angle of gradient, perform non-maxima supression.
  // fill the map with one of the following values:
  //   0 - the pixel might belong to an edge
  //   1 - the pixel can not belong to an edge
  //   2 - the pixel does belong to an edge
  for( i = 0; i <= size.height; i++ )
  {
    int* _mag    = mag_buf[( i > 0 ) + 1] + 1;
    float* _magf = (float*)_mag;
    const float* _dx = (float*)( dx->data.ptr + dx->step * i );
    const float* _dy = (float*)( dy->data.ptr + dy->step * i );
    uchar* _map;
    int x, y;
    ptrdiff_t magstep1, magstep2;
    int prev_flag = 0;

    if( i < size.height )
    {
      _mag[-1] = _mag[size.width] = 0;

      if( !( flags & CV_CANNY_L2_GRADIENT ) )
        for( j = 0; j < size.width; j++ )
          _mag[j] = std::abs( _dx[j] ) + std::abs( _dy[j] );

      else
      {
        for( j = 0; j < size.width; j++ )
        {
          x        = int( _dx[j] );
          y = int ( _dy[j] );
          _magf[j] = (float)std::sqrt( (double)x * x + (double)y * y );
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
    _dx = (float*)( dx->data.ptr + dx->step * ( i - 1 ) );
    _dy = (float*)( dy->data.ptr + dy->step * ( i - 1 ) );

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

  ROM_COUT_LILIAN( "Canny 2 took : " << t.elapsed() );

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

  ROM_COUT_LILIAN( "Canny 3 took : " << t.elapsed() );

  t.restart();

  // the final pass, form the final image
  for( i = 0; i < size.height; i++ )
  {
    const uchar* _map = map + mapstep * ( i + 1 ) + 1;
    uchar* _dst       = dst->data.ptr + dst->step * i;

    //const short* _dx = (short*)(dx->data.ptr + dx->step*i);
    //const short* _dy = (short*)(dy->data.ptr + dy->step*i);

    for( j = 0; j < size.width; j++ )
    {
      _dst[j] = ( uchar ) - ( _map[j] >> 1 );

      /*if(_dst[j])
         {
          label->push_back(popart::vision::EdgePoint(j, i, _dx[j], _dy[j], label));
          popart::vision::EdgePoint * p = & label->back() ;
          p->_label = label;
          labelsMap[j][i] = p;
         }*/

    }

  }
  ROM_COUT_LILIAN( "Canny 4 : " << t.elapsed() );
}

#endif //with_cuda


void cvRecodedCanny(
  void* srcarr,
  void* dstarr,
  CvMat*& dx,
  CvMat*& dy,
  double low_thresh,
  double high_thresh,
  int aperture_size )
{
  boost::timer t;  
  cv::AutoBuffer<char> buffer;
  std::vector<uchar*> stack;
  uchar** stack_top = 0, ** stack_bottom = 0;

  CvMat srcstub, * src = cvGetMat( srcarr, &srcstub );
  CvMat dststub, * dst = cvGetMat( dstarr, &dststub );
  CvSize size;
  int flags = aperture_size;
  int low, high;
  int* mag_buf[3];
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
    double t;
    CV_SWAP( low_thresh, high_thresh, t );
  }

  aperture_size &= INT_MAX;
  if( ( aperture_size & 1 ) == 0 || aperture_size < 3 || aperture_size > 7 )
    CV_Error( CV_StsBadFlag, "" );

  size = cvGetMatSize( src );

  // TODO: no allocation here:
  dx = cvCreateMat( size.height, size.width, CV_16SC1 );
  dy = cvCreateMat( size.height, size.width, CV_16SC1 );

  // cvSobel is the function called by default in OpenCV, with a 3x3 kernel
  // to compute the derivative in x and y.
  // The kernel used to compute the derivative is changed here by a 9x9 one, to stick
  // with the results obtained with the canny implementation in the Matlab image
  // processing toolbox (2012)

//	{
//		dx = cvCreateMat( size.height, size.width, CV_32FC1 );
//		dy = cvCreateMat( size.height, size.width, CV_32FC1 );
//	}

  ROM_COUT_LILIAN( "Sobel allocation : " << t.elapsed() );
  t.restart();

  //cvSobel( src, dx, 1, 0, aperture_size );
  //cvSobel( src, dy, 0, 1, aperture_size );

  {
    CvMat* kerneldX = cvCreateMat( 9, 9, CV_32FC1 );
    CvMat* kerneldY = cvCreateMat( 9, 9, CV_32FC1 );

    CV_MAT_ELEM( *kerneldX, float, 0, 0 ) = 0.000000143284235  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 1 ) = 0.000003558691641  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 2 ) = 0.000028902492951  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 3 ) = 0.000064765993382  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 5 ) = -0.000064765993382  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 6 ) = -0.000028902492951  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 7 ) = -0.000003558691641  ;
    CV_MAT_ELEM( *kerneldX, float, 0, 8 ) = -0.000000143284235  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 0 ) = 0.000004744922188  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 1 ) = 0.000117847682078  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 2 ) = 0.000957119116802  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 3 ) = 0.002144755142391  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 5 ) = -0.002144755142391  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 6 ) = -0.000957119116802  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 7 ) = -0.000117847682078  ;
    CV_MAT_ELEM( *kerneldX, float, 1, 8 ) = -0.000004744922188  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 0 ) = 0.000057804985902  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 1 ) = 0.001435678675203  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 2 ) = 0.011660097860113  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 3 ) = 0.026128466569370  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 5 ) = -0.026128466569370  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 6 ) = -0.011660097860113  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 7 ) = -0.001435678675203  ;
    CV_MAT_ELEM( *kerneldX, float, 2, 8 ) = -0.000057804985902  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 0 ) = 0.000259063973527  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 1 ) = 0.006434265427174  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 2 ) = 0.052256933138740  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 3 ) = 0.117099663048638  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 5 ) = -0.117099663048638  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 6 ) = -0.052256933138740  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 7 ) = -0.006434265427174  ;
    CV_MAT_ELEM( *kerneldX, float, 3, 8 ) = -0.000259063973527  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 0 ) = 0.000427124283626  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 1 ) = 0.010608310271112  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 2 ) = 0.086157117207395  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 3 ) = 0.193064705260108  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 5 ) = -0.193064705260108  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 6 ) = -0.086157117207395  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 7 ) = -0.010608310271112  ;
    CV_MAT_ELEM( *kerneldX, float, 4, 8 ) = -0.000427124283626  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 0 ) = 0.000259063973527  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 1 ) = 0.006434265427174  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 2 ) = 0.052256933138740  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 3 ) = 0.117099663048638  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 5 ) = -0.117099663048638  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 6 ) = -0.052256933138740  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 7 ) = -0.006434265427174  ;
    CV_MAT_ELEM( *kerneldX, float, 5, 8 ) = -0.000259063973527  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 0 ) = 0.000057804985902  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 1 ) = 0.001435678675203  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 2 ) = 0.011660097860113  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 3 ) = 0.026128466569370  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 5 ) = -0.026128466569370  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 6 ) = -0.011660097860113  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 7 ) = -0.001435678675203  ;
    CV_MAT_ELEM( *kerneldX, float, 6, 8 ) = -0.000057804985902  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 0 ) = 0.000004744922188  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 1 ) = 0.000117847682078  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 2 ) = 0.000957119116802  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 3 ) = 0.002144755142391  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 5 ) = -0.002144755142391  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 6 ) = -0.000957119116802  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 7 ) = -0.000117847682078  ;
    CV_MAT_ELEM( *kerneldX, float, 7, 8 ) = -0.000004744922188  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 0 ) = 0.000000143284235  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 1 ) = 0.000003558691641  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 2 ) = 0.000028902492951  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 3 ) = 0.000064765993382  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 4 ) = 0  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 5 ) = -0.000064765993382  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 6 ) = -0.000028902492951  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 7 ) = -0.000003558691641  ;
    CV_MAT_ELEM( *kerneldX, float, 8, 8 ) = -0.000000143284235  ;

    cvConvertScale( kerneldX, kerneldX, -1.f );
    cvTranspose( kerneldX, kerneldY );

    cvFilter2D( src, dx, kerneldX );
    cvFilter2D( src, dy, kerneldY );

    cvReleaseMat( &kerneldX );
    cvReleaseMat( &kerneldY );
  }

  if( flags & CV_CANNY_L2_GRADIENT )
  {
    Cv32suf ul, uh;
    ul.f = (float)low_thresh;
    uh.f = (float)high_thresh;

    low  = ul.i;
    high = uh.i;
  }
  else
  {
    low  = cvFloor( low_thresh );
    high = cvFloor( high_thresh );
  }

  buffer.allocate( ( size.width + 2 ) * ( size.height + 2 ) + ( size.width + 2 ) * 3 * sizeof( int ) );

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

  ROM_COUT_DEBUG( "Canny 1 took: " << t.elapsed() );
  t.restart();

  // calculate magnitude and angle of gradient, perform non-maxima supression.
  // fill the map with one of the following values:
  //   0 - the pixel might belong to an edge
  //   1 - the pixel can not belong to an edge
  //   2 - the pixel does belong to an edge
  for( i = 0; i <= size.height; i++ )
  {
    int* _mag    = mag_buf[( i > 0 ) + 1] + 1;
    float* _magf = (float*)_mag;
    const short* _dx = (short*)( dx->data.ptr + dx->step * i );
    const short* _dy = (short*)( dy->data.ptr + dy->step * i );
    uchar* _map;
    int x, y;
    ptrdiff_t magstep1, magstep2;
    int prev_flag = 0;

    if( i < size.height )
    {
      _mag[-1] = _mag[size.width] = 0;

      if( !( flags & CV_CANNY_L2_GRADIENT ) )
        for( j = 0; j < size.width; j++ )
          _mag[j] = abs( _dx[j] ) + abs( _dy[j] );

      else
      {
        for( j = 0; j < size.width; j++ )
        {
          x        = _dx[j];
          y = _dy[j];
          _magf[j] = (float)std::sqrt( (double)x * x + (double)y * y );
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

  ROM_COUT_DEBUG( "Canny 2 took : " << t.elapsed() );

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

  ROM_COUT_DEBUG( "Canny 3 took : " << t.elapsed() );

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
  ROM_COUT_DEBUG( "Canny 4 : " << t.elapsed() );
}
