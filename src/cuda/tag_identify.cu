/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cuda_runtime.h>

#include "cuda/tag.h"
#include "cuda/frame.h"
#include "cuda/frameparam.h"
#include "cuda/clamp.h"
#include "cuda/assist.h"
#include "cuda/geom_matrix.h"
#include "cuda/nearby_point.h"
#include "cuda/tag_cut.h"

#if 1
    __device__ __host__
    inline int validate( const char* file, int line, int input, int reference )
    {
#if 1
        return min( input, reference );
#else
        if( input < reference ) {
            // printf( "%s:%d Divergence: run-time value %d < conf %d\n", file, line, input, reference );
            return input;
        }
        if( input > reference ) {
            // printf( "%s:%d Divergence: run-time value %d > conf %d\n", file, line, input, reference );
            return reference;
        }
        return reference;
#endif
    }
    #define STRICT_CUTSIZE(sz) validate( __FILE__, __LINE__, sz, 22 )
    #define STRICT_SAMPLE(sz)  validate( __FILE__, __LINE__, sz, 5 )
    #define STRICT_SIGSIZE(sz) validate( __FILE__, __LINE__, sz, 100 )
#else
    #define STRICT_CUTSIZE(sz) sz
    #define STRICT_SAMPLE(sz)  sz
    #define STRICT_SIGSIZE(sz) sz
#endif

using namespace std;

namespace popart {

namespace identification {

__device__
inline
float getPixelBilinear( const cv::cuda::PtrStepSzb src, float2 xy )
{
    const int px = clamp( (int)xy.x, src.cols ); // floor of x
    const int py = clamp( (int)xy.y, src.rows ); // floor of y

    // uint8_t p0 = src.ptr(py  )[px  ];
    const uint8_t p1 = src.ptr(py  )[px  ];
    const uint8_t p2 = src.ptr(py  )[px+1];
    const uint8_t p3 = src.ptr(py+1)[px  ];
    const uint8_t p4 = src.ptr(py+1)[px+1];

    // Calculate the weights for each pixel
    const float fx  = xy.x - (float)px;
    const float fy  = xy.y - (float)py;
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;

    const float w1 = fx1 * fy1;
    const float w2 = fx  * fy1;
    const float w3 = fx1 * fy;
    const float w4 = fx  * fy;

    // Calculate the weighted sum of pixels (for each color channel)
    return ( p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4 ) / 2.0f;
}

__device__
inline
void extractSignalUsingHomography( const CutStruct&                   cut,
                                   CutSignals&                        signals,
                                   const cv::cuda::PtrStepSzb         src,
                                   const popart::geometry::matrix3x3& mHomography,
                                   const popart::geometry::matrix3x3& mInvHomography )
{
    float2 backProjStop;

    backProjStop = mInvHomography.applyHomography( cut.stop );
  
    const float xStart = backProjStop.x * cut.beginSig;
    const float yStart = backProjStop.y * cut.beginSig;
    const float xStop  = backProjStop.x * cut.endSig; // xStop and yStop must not be normalised but the 
    const float yStop  = backProjStop.y * cut.endSig; // norm([xStop;yStop]) is supposed to be close to 1.

    // Compute the steps stepX and stepY along x and y.
    const int   nSamples = STRICT_SIGSIZE(cut.sigSize);
    const float stepX = ( xStop - xStart ) / ( STRICT_SIGSIZE(nSamples) - 1.0f );
    const float stepY = ( yStop - yStart ) / ( STRICT_SIGSIZE(nSamples) - 1.0f );
    const float stepX32 = 32.0f * stepX;
    const float stepY32 = 32.0f * stepY;

    // float x =  xStart; - serial code
    // float y =  yStart; - serial code
    float x =  xStart + threadIdx.x * stepX;
    float y =  yStart + threadIdx.x * stepY;
    for( std::size_t i = threadIdx.x; i < STRICT_SIGSIZE(nSamples); i += 32 ) {
        float2 xyRes;

        // [xRes;yRes;1] ~= mHomography*[x;y;1.0]
        xyRes = mHomography.applyHomography( x, y );

        bool breaknow = ( xyRes.x < 1.0f && xyRes.x > src.cols-1 && xyRes.y < 1.0f && xyRes.y > src.rows-1 );

        if( __any( breaknow ) )
        {
            if( threadIdx.x == 0 ) signals.outOfBounds = 1;
            return;
        }

        // Bilinear interpolation
        signals.sig[i] = popart::identification::getPixelBilinear( src, xyRes );

        x += stepX32;
        y += stepY32;
    }
}


/* We use a pair of homographies to extract signals (<=128) for every
 * cut, and every nearby point of a potential center.
 * vCutMaxVecLen is a safety variable, could actually be 100.
 * The function is called from
 *     idNearbyPointDispatcher
 * once for every
 */
__global__
void idGetSignals( cv::cuda::PtrStepSzb   src,
                   TagPipe::ImageCenter*  d_image_center_opt_input,
                   const NearbyPointGrid* point_grid,
                   const CutStructGrid*   cut_grid,
                   CutSignalGrid*         sig_grid )
{
    TagPipe::ImageCenter& v = d_image_center_opt_input[blockIdx.x];
    if( not v._valid ) return;
    if( v._iterations <= 0 ) return;

    const int              vCutSize     = v._vCutSize;
    const NearbyPointGrid* point_buffer = &point_grid[blockIdx.x];
    const CutStructGrid*   cut_buffer   = &cut_grid[blockIdx.x];
    CutSignalGrid*         sig_buffer   = &sig_grid[blockIdx.x];

    if( threadIdx.y >= vCutSize ) return;

    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);
    const int i = blockIdx.y;
    const int j = blockIdx.z;
    const int nearbyPointIndex = j * STRICT_SAMPLE(gridNSample) + i;
    const int cutIndex = threadIdx.y;

    const NearbyPoint& nPoint = point_buffer->getGrid(i,j);

    if( cutIndex >= STRICT_CUTSIZE(vCutSize) ) {
        // warps do never loose lockstep because of this
        return; // out of bounds
    }

    const CutStruct& myCut = cut_buffer->getGrid( cutIndex );
    CutSignals& mySignals  = sig_buffer->getGrid( cutIndex, i, j ); //[nearbyPointIndex * STRICT_CUTSIZE(vCutSize) + cutIndex];

    if( threadIdx.x == 0 ) mySignals.outOfBounds = 0;

    extractSignalUsingHomography( myCut,
                                  mySignals,
                                  src,
                                  nPoint.mHomography,
                                  nPoint.mInvHomography );
}

/* We use a pair of homographies to extract signals (<=128) for every
 * cut, and every nearby point of a potential center.
 * vCutMaxVecLen is a safety variable, could actually be 100.
 * The function is called from
 *     idNearbyPointDispatcher
 * once for every
 */
__global__
void idGetSignals( cv::cuda::PtrStepSzb   src,
                   const int              vCutSize,
                   const NearbyPointGrid* point_buffer,
                   const CutStructGrid*   cut_buffer,
                   CutSignalGrid*         sig_buffer )
{
    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);
    const int i = blockIdx.y;
    const int j = blockIdx.z;
    const int nearbyPointIndex = j * STRICT_SAMPLE(gridNSample) + i;
    const int cutIndex = threadIdx.y;

    const NearbyPoint& nPoint = point_buffer->getGrid(i,j);

    if( cutIndex >= STRICT_CUTSIZE(vCutSize) ) {
        // warps do never loose lockstep because of this
        return; // out of bounds
    }

    const CutStruct& myCut = cut_buffer->getGrid( cutIndex );
    CutSignals& mySignals  = sig_buffer->getGrid( cutIndex, i, j ); //[nearbyPointIndex * STRICT_CUTSIZE(vCutSize) + cutIndex];

    if( threadIdx.x == 0 ) mySignals.outOfBounds = 0;

    extractSignalUsingHomography( myCut,
                                  mySignals,
                                  src,
                                  nPoint.mHomography,
                                  nPoint.mInvHomography );
}

__global__
void initAllNearbyPoints(
    bool                               first_iteration,
    TagPipe::ImageCenter*              d_image_center_opt_input,
    const float                        currentNeighbourSize,
    NearbyPointGrid*                   nearbyPoints )
{
    TagPipe::ImageCenter& v = d_image_center_opt_input[blockIdx.x];
    if( not v._valid ) return;
    if( v._iterations <= 0 ) return;

    const float neighbourSize = currentNeighbourSize * v._transformedEllipseMaxRadius;

    const popart::geometry::ellipse&   ellipse = v._outerEllipse;
    const popart::geometry::matrix3x3& mT      = v._mT;
    float2                             center  = v._center;
    NearbyPointGrid*         d_nearbyPointGrid = &nearbyPoints[blockIdx.x];

    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);

    assert( gridDim.y == STRICT_SAMPLE(gridNSample) );
    assert( gridDim.z == STRICT_SAMPLE(gridNSample) );

    const float  gridWidth   = neighbourSize;
    const float  halfWidth   = gridWidth/2.0f;
    const float  stepSize    = gridWidth * __frcp_rn( float(STRICT_SAMPLE(gridNSample)-1) );

    if( not first_iteration ) {
        // the best center is located in point_buffer[0]
        // center = point_buffer[0].point;
        center = d_nearbyPointGrid->getGrid(0,0).point;
        // @lilian: why do we need this "center = mT * center" in every iteration?
    }
    mT.condition( center );

    const int i = blockIdx.y;
    const int j = blockIdx.z;
    // const int idx = j * STRICT_SAMPLE(gridNSample) + i;

    NearbyPoint& nPoint = d_nearbyPointGrid->getGrid(i,j);

    float2 condCenter = make_float2( center.x - halfWidth + i*stepSize,
                                     center.y - halfWidth + j*stepSize );

    popart::geometry::matrix3x3  mInvT;
    mT.invert( mInvT ); // note: returns false if it fails
    mInvT.condition( condCenter );

    nPoint.point    = condCenter;
    nPoint.result   = 0.0f;
    nPoint.resSize  = 0;
    nPoint.readable = true;
    ellipse.computeHomographyFromImagedCenter( nPoint.point, nPoint.mHomography );
    nPoint.mHomography.invert( nPoint.mInvHomography );
}

#if 1
/* All the signals have been computed for the homographies rolled
 * this NearbyPoint structure by the previous call to idGetSignals.
 * Now we want to compare all possible combinations of Cuts and
 * select the one that has minimizes the signal.
 * Essential a full matrix of checks, but one triangle excluding the
 * diagonal of the matrix is sufficient.
 * We use 32 threads per NearbyPoint, and threads must find the
 * minimum for ceil(#Cuts/32) first, before finding using shuffle to
 * store the minimum in result in the NearbyPoint structure.
 */
__global__
void idComputeResult( TagPipe::ImageCenter* d_image_center_opt_input,
                      NearbyPointGrid*      point_grid,
                      const CutStructGrid*  cut_grid,
                      const CutSignalGrid*  sig_grid )
{
    TagPipe::ImageCenter& v = d_image_center_opt_input[blockIdx.z];
    if( not v._valid ) return;
    if( v._iterations <= 0 ) return;

    const int              vCutSize          = v._vCutSize;
    NearbyPointGrid*       d_NearbyPointGrid = &point_grid[blockIdx.z];
    const CutStructGrid*   cut_buffer        = &cut_grid[blockIdx.z];
    const CutSignalGrid*   sig_buffer        = &sig_grid[blockIdx.z];

    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);
    const int grid_i   = blockIdx.y % gridNSample;
    const int grid_j   = blockIdx.y / gridNSample;
    // const int grid_idx = blockIdx.y; // grid_j * gridNSample + grid_i;
    // const CutSignals* allcut_signals = &sig_buffer[grid_idx * STRICT_CUTSIZE(vCutSize)];
    NearbyPoint& nPoint  = d_NearbyPointGrid->getGrid( grid_i, grid_j );

    int myPair = blockIdx.x * 32 + threadIdx.y;
    int left_cut  = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
    int right_cut = myPair - left_cut*(left_cut-1)/2;

    int   ct   = 0;
    float val  = 0.0f;
    bool  comp = true;

    comp = ( left_cut < STRICT_CUTSIZE(vCutSize) && right_cut < left_cut );

    if( comp ) {
        // const CutSignals* l_signals = &allcut_signals[i];
        // const CutSignals* r_signals = &allcut_signals[left_cut];
        const CutSignals& l_signals = sig_buffer->getGrid( left_cut,  grid_i, grid_j );
        const CutSignals& r_signals = sig_buffer->getGrid( right_cut, grid_i, grid_j );
        comp  = ( threadIdx.x < tagParam.sampleCutLength ) &&
                  not l_signals.outOfBounds &&
                  not r_signals.outOfBounds;
        if( comp ) {
            const int limit = STRICT_SIGSIZE(cut_buffer->getGrid(left_cut).sigSize); // we could also use right_cut
            for( int offset = threadIdx.x; offset < STRICT_SIGSIZE(limit); offset += 32 ) {
                float square = l_signals.sig[offset] - r_signals.sig[offset];
                val += ( square * square );
            }
            ct = 1;
        }
    }

    val += __shfl_down( val, 16 );
    val += __shfl_down( val,  8 );
    val += __shfl_down( val,  4 );
    val += __shfl_down( val,  2 );
    val += __shfl_down( val,  1 );

    __shared__ float signal_sum[32];
    __shared__ int   count_sum[32];

    if( threadIdx.x == 0 ) {
        signal_sum[threadIdx.y] = val;
        count_sum [threadIdx.y] = ct;
    }

    __syncthreads();

    if( threadIdx.y == 0 ) {
        val = signal_sum[threadIdx.x];
        val += __shfl_down( val, 16 );
        val += __shfl_down( val,  8 );
        val += __shfl_down( val,  4 );
        val += __shfl_down( val,  2 );
        val += __shfl_down( val,  1 );
        ct  = count_sum[threadIdx.x];
        ct  += __shfl_down( ct, 16 );
        ct  += __shfl_down( ct,  8 );
        ct  += __shfl_down( ct,  4 );
        ct  += __shfl_down( ct,  2 );
        ct  += __shfl_down( ct,  1 );

        if( threadIdx.x == 0 ) {
            atomicAdd( &nPoint.result,  val );
            atomicAdd( &nPoint.resSize, ct );
        }
    }
}
#else
/* All the signals have been computed for the homographies rolled
 * this NearbyPoint structure by the previous call to idGetSignals.
 * Now we want to compare all possible combinations of Cuts and
 * select the one that has minimizes the signal.
 * Essential a full matrix of checks, but one triangle excluding the
 * diagonal of the matrix is sufficient.
 * We use 32 threads per NearbyPoint, and threads must find the
 * minimum for ceil(#Cuts/32) first, before finding using shuffle to
 * store the minimum in result in the NearbyPoint structure.
 */
__global__
void idComputeResult( NearbyPointGrid*     d_NearbyPointGrid,
                      const CutStructGrid* cut_buffer,
                      const CutSignalGrid* sig_buffer,
                      const int            vCutSize )
{
    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);
    const int grid_i   = blockIdx.y;
    const int grid_j   = blockIdx.z;
    const int grid_idx = grid_j * STRICT_SAMPLE(gridNSample) + grid_i;
    // const CutSignals* allcut_signals = &sig_buffer[grid_idx * STRICT_CUTSIZE(vCutSize)];
    NearbyPoint& nPoint  = d_NearbyPointGrid->getGrid( grid_i, grid_j );

    int myPair = blockIdx.x * 32 + threadIdx.y;
    int left_cut  = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
    int right_cut = myPair - left_cut*(left_cut-1)/2;

    int   ct   = 0;
    float val  = 0.0f;
    bool  comp = true;

    comp = ( left_cut < STRICT_CUTSIZE(vCutSize) && right_cut < left_cut );

    if( comp ) {
        // const CutSignals* l_signals = &allcut_signals[i];
        // const CutSignals* r_signals = &allcut_signals[left_cut];
        const CutSignals& l_signals = sig_buffer->getGrid( left_cut,  grid_i, grid_j );
        const CutSignals& r_signals = sig_buffer->getGrid( right_cut, grid_i, grid_j );
        comp  = ( threadIdx.x < tagParam.sampleCutLength ) &&
                  not l_signals.outOfBounds &&
                  not r_signals.outOfBounds;
        if( comp ) {
            const int limit = STRICT_SIGSIZE(cut_buffer->getGrid(left_cut).sigSize); // we could also use right_cut
            for( int offset = threadIdx.x; offset < STRICT_SIGSIZE(limit); offset += 32 ) {
                float square = l_signals.sig[offset] - r_signals.sig[offset];
                val += ( square * square );
            }
            ct = 1;
        }
    }

    val += __shfl_down( val, 16 );
    val += __shfl_down( val,  8 );
    val += __shfl_down( val,  4 );
    val += __shfl_down( val,  2 );
    val += __shfl_down( val,  1 );

    __shared__ float signal_sum[32];
    __shared__ int   count_sum[32];

    if( threadIdx.x == 0 ) {
        signal_sum[threadIdx.y] = val;
        count_sum [threadIdx.y] = ct;
    }

    __syncthreads();

    if( threadIdx.y == 0 ) {
        val = signal_sum[threadIdx.x];
        val += __shfl_down( val, 16 );
        val += __shfl_down( val,  8 );
        val += __shfl_down( val,  4 );
        val += __shfl_down( val,  2 );
        val += __shfl_down( val,  1 );
        ct  = count_sum[threadIdx.x];
        ct  += __shfl_down( ct, 16 );
        ct  += __shfl_down( ct,  8 );
        ct  += __shfl_down( ct,  4 );
        ct  += __shfl_down( ct,  2 );
        ct  += __shfl_down( ct,  1 );

        if( threadIdx.x == 0 ) {
            atomicAdd( &nPoint.result,  val );
            atomicAdd( &nPoint.resSize, ct );
        }
    }
}
#endif

#if 1
__global__
void idBestNearbyPoint32plus( TagPipe::ImageCenter* d_image_center_opt_input,
                              NearbyPointGrid*      point_grid )
{
    TagPipe::ImageCenter& v = d_image_center_opt_input[blockIdx.x];
    if( not v._valid ) return;
    if( v._iterations <= 0 ) return;

    NearbyPointGrid*       d_NearbyPointGrid = &point_grid[blockIdx.x];

    // phase 1: each thread searches for its own best point
    float bestRes = FLT_MAX;
    const int gridNSample = tagParam.gridNSample;
    const int   gridSquare = gridNSample * gridNSample;
    int   bestIdx = gridSquare-1;
    int   idx;
    for( idx=threadIdx.x; idx<gridSquare; idx+=32 ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            bestIdx = idx;
            bestRes = point.result / point.resSize;
            break;
        }
    }
    __syncthreads();
    for( ; idx<gridSquare; idx+=32 ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            float val = point.result / point.resSize;
            if( val < bestRes ) {
                bestIdx = idx;
                bestRes = val;
            }
        }
    }
    __syncthreads();

    // phase 2: reduce to let thread 0 know the best point
    #pragma unroll
    for( int shft=4; shft>=0; shft-- ) {
        int otherRes = __shfl_down( bestRes, (1 << shft) );
        int otherIdx = __shfl_down( bestIdx, (1 << shft) );
        if( otherRes < bestRes ) {
            bestRes = otherRes;
            bestIdx = otherIdx;
        }
    }
    __syncthreads();

    // phase 3: copy the best point into index 0
    if( threadIdx.x == 0 ) {
        if( bestIdx != 0 ) {
            const int x = bestIdx % gridNSample;
            const int y = bestIdx / gridNSample;
            const NearbyPoint& src_point = d_NearbyPointGrid->getGrid(x,y);
            NearbyPoint&       dst_point = d_NearbyPointGrid->getGrid(0,0);
            memcpy( &dst_point, &src_point, sizeof( NearbyPoint ) );
            dst_point.residual = bestRes;
        }
    }
}
#else
__global__
void idBestNearbyPoint32plus( NearbyPointGrid* d_NearbyPointGrid,
                              const size_t     gridNSample )
{
    // phase 1: each thread searches for its own best point
    float bestRes = FLT_MAX;
    int   gridSquare = gridNSample * gridNSample;
    int   bestIdx = gridSquare-1;
    int   idx;
    for( idx=threadIdx.x; idx<gridSquare; idx+=32 ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            bestIdx = idx;
            bestRes = point.result / point.resSize;
            break;
        }
    }
    __syncthreads();
    for( ; idx<gridSquare; idx+=32 ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            float val = point.result / point.resSize;
            if( val < bestRes ) {
                bestIdx = idx;
                bestRes = val;
            }
        }
    }
    __syncthreads();

    // phase 2: reduce to let thread 0 know the best point
    #pragma unroll
    for( int shft=4; shft>=0; shft-- ) {
        int otherRes = __shfl_down( bestRes, (1 << shft) );
        int otherIdx = __shfl_down( bestIdx, (1 << shft) );
        if( otherRes < bestRes ) {
            bestRes = otherRes;
            bestIdx = otherIdx;
        }
    }
    __syncthreads();

    // phase 3: copy the best point into index 0
    if( threadIdx.x == 0 ) {
        if( bestIdx != 0 ) {
            const int x = bestIdx % gridNSample;
            const int y = bestIdx / gridNSample;
            const NearbyPoint& src_point = d_NearbyPointGrid->getGrid(x,y);
            NearbyPoint&       dst_point = d_NearbyPointGrid->getGrid(0,0);
            memcpy( &dst_point, &src_point, sizeof( NearbyPoint ) );
            dst_point.residual = bestRes;
        }
    }
}
#endif

#if 1
__global__
void idBestNearbyPoint31max( TagPipe::ImageCenter* d_image_center_opt_input,
                             NearbyPointGrid*      point_grid )
{
    TagPipe::ImageCenter& v = d_image_center_opt_input[blockIdx.x];
    if( not v._valid ) return;
    if( v._iterations <= 0 ) return;

    NearbyPointGrid*       d_NearbyPointGrid = &point_grid[blockIdx.x];

    // phase 1: each thread retrieves its point
    const int gridNSample = tagParam.gridNSample;
    const size_t gridSquare = gridNSample * gridNSample;
    float bestRes = FLT_MAX;
    int   bestIdx = gridSquare-1;
    int   idx     = threadIdx.x;
    if( idx < gridSquare ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            bestIdx = idx;
            bestRes = point.result / point.resSize;
        }
    }
    __syncthreads();

    // phase 2: reduce to let thread 0 know the best point
    #pragma unroll
    for( int shft=4; shft>=0; shft-- ) {
        int otherRes = __shfl_down( bestRes, (1 << shft) );
        int otherIdx = __shfl_down( bestIdx, (1 << shft) );
        if( otherRes < bestRes ) {
            bestRes = otherRes;
            bestIdx = otherIdx;
        }
    }
    __syncthreads();

    // phase 3: copy the best point into index 0
    if( threadIdx.x == 0 ) {
        if( bestIdx != 0 ) {
            const int x = bestIdx % gridNSample;
            const int y = bestIdx / gridNSample;
            const NearbyPoint& src_point = d_NearbyPointGrid->getGrid(x,y);
            NearbyPoint&       dst_point = d_NearbyPointGrid->getGrid(0,0);
            memcpy( &dst_point, &src_point, sizeof( NearbyPoint ) );
            dst_point.residual = bestRes;
        }
    }
}
#else
__global__
void idBestNearbyPoint31max( NearbyPointGrid* d_NearbyPointGrid,
                             const size_t     gridNSample )
{
    // phase 1: each thread retrieves its point
    const size_t gridSquare = gridNSample * gridNSample;
    float bestRes = FLT_MAX;
    int   bestIdx = gridSquare-1;
    int   idx     = threadIdx.x;
    if( idx < gridSquare ) {
        const int x = idx % gridNSample;
        const int y = idx / gridNSample;
        const NearbyPoint& point = d_NearbyPointGrid->getGrid(x,y);
        if( point.readable ) {
            bestIdx = idx;
            bestRes = point.result / point.resSize;
        }
    }
    __syncthreads();

    // phase 2: reduce to let thread 0 know the best point
    #pragma unroll
    for( int shft=4; shft>=0; shft-- ) {
        int otherRes = __shfl_down( bestRes, (1 << shft) );
        int otherIdx = __shfl_down( bestIdx, (1 << shft) );
        if( otherRes < bestRes ) {
            bestRes = otherRes;
            bestIdx = otherIdx;
        }
    }
    __syncthreads();

    // phase 3: copy the best point into index 0
    if( threadIdx.x == 0 ) {
        if( bestIdx != 0 ) {
            const int x = bestIdx % gridNSample;
            const int y = bestIdx / gridNSample;
            const NearbyPoint& src_point = d_NearbyPointGrid->getGrid(x,y);
            NearbyPoint&       dst_point = d_NearbyPointGrid->getGrid(0,0);
            memcpy( &dst_point, &src_point, sizeof( NearbyPoint ) );
            dst_point.residual = bestRes;
        }
    }
}
#endif

} // namespace identification

/**
 * @pre the cuts for this tag have been uploaded.
 * @param[in] i the index in the ImageCenter vector
 * @param[in] iterations the caller defines how many refinement loop we execute
 */
__host__
void TagPipe::idCostFunction( )
{
    const size_t gridNSample = STRICT_SAMPLE(_params._imagedCenterNGridSample);

    int iterations = 0;
    for( int i=0; i<_num_cut_struct_grid; i++ ) {
        ImageCenter& v = _h_image_center_opt_input[i];
        if( v._valid ) {
            iterations = std::max( iterations, v._iterations );
        }
    }

    // NOTE: with a numTags parameter we could copy a bit faster
    POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( _d_image_center_opt_input,
                                     _h_image_center_opt_input,
                                     _num_cut_struct_grid * sizeof(ImageCenter),
                                     _tag_stream );

    bool first_iteration = true;

    float currentNeighbourSize = _params._imagedCenterNeighbourSize;

    for( ; iterations>0; iterations-- ) {
        dim3 block( 1, 1, 1 );
        dim3 grid( _num_cut_struct_grid, gridNSample, gridNSample );

        popart::identification::initAllNearbyPoints
            <<<grid,block,0,_tag_stream>>>
            ( first_iteration,
              _d_image_center_opt_input,
              currentNeighbourSize,
              _d_nearby_point_grid );

#if 1
        dim3 get_block( 32, _params._numCutsInIdentStep, 1 ); // we use this to sum up signals
        dim3 get_grid( _num_cut_struct_grid, gridNSample, gridNSample );

        popart::identification::idGetSignals
            <<<get_grid,get_block,0,_tag_stream>>>
            ( _frame[0]->getPlaneDev(),
              _d_image_center_opt_input,
              _d_nearby_point_grid,
              _d_cut_struct_grid,
              _d_cut_signal_grid );
#else
        for( int i=0; i<_num_cut_struct_grid; i++ ) {
            ImageCenter& v = _h_image_center_opt_input[i];
            if( not v._valid ) continue;
            if( v._iterations <= 0 ) continue;

            dim3 get_block( 32, STRICT_CUTSIZE(v._vCutSize), 1 ); // we use this to sum up signals
            dim3 get_grid( 1, gridNSample, gridNSample );

            popart::identification::idGetSignals
                <<<get_grid,get_block,0,_tag_stream>>>
                ( _frame[0]->getPlaneDev(),
                  STRICT_CUTSIZE(v._vCutSize),
                  getNearbyPointGridBuffer( v._tagIndex ),        // in
                  getCutStructGridBufferDev( v._tagIndex ),
                  getSignalGridBuffer( v._tagIndex ) );
        }
#endif

#if 1
            dim3 id_block( 32, // we use this to sum up signals
                        32, // we can use some shared memory/warp magic for summing
                        1 );
            const int numPairs = _params._numCutsInIdentStep * (_params._numCutsInIdentStep-1) / 2;
            dim3 id_grid( grid_divide( numPairs, 32 ),
                          gridNSample*gridNSample,
                          _num_cut_struct_grid );

            popart::identification::idComputeResult
                <<<id_grid,id_block,0,_tag_stream>>>
                ( _d_image_center_opt_input,
                  _d_nearby_point_grid,
                  _d_cut_struct_grid,
                  _d_cut_signal_grid );
#else
        for( int i=0; i<_num_cut_struct_grid; i++ ) {
            ImageCenter& v = _h_image_center_opt_input[i];
            if( not v._valid ) continue;
            if( v._iterations <= 0 ) continue;

            dim3 id_block( 32, // we use this to sum up signals
                        32, // we can use some shared memory/warp magic for summing
                        1 );
            const int numPairs = STRICT_CUTSIZE(v._vCutSize)*(STRICT_CUTSIZE(v._vCutSize)-1)/2;
            dim3 id_grid( grid_divide( numPairs, 32 ),
                          gridNSample,
                          gridNSample );

            popart::identification::idComputeResult
                <<<id_grid,id_block,0,_tag_stream>>>
                ( getNearbyPointGridBuffer( v._tagIndex ),
                  getCutStructGridBufferDev( v._tagIndex ),
                  getSignalGridBuffer( v._tagIndex ),
                  STRICT_CUTSIZE(v._vCutSize) );
        }
#endif

#if 1
        /* We search for the minimum of gridNSample x gridNSample
         * nearby points. Default for gridNSample is 5.
         * It is therefore most efficient to use a single-warp kernel
         * for the search.
         */
        const int gridSquare = gridNSample * gridNSample;

        if( gridSquare < 32 ) {
            popart::identification::idBestNearbyPoint31max
                <<<_num_cut_struct_grid,32,0,_tag_stream>>>
                ( _d_image_center_opt_input,
                  _d_nearby_point_grid );
        } else {
            cerr << __FILE__ << ":" << __LINE__
                << " Untested code idBestNearbyPoint32plus" << endl;
            popart::identification::idBestNearbyPoint32plus
                <<<_num_cut_struct_grid,32,0,_tag_stream>>>
                ( _d_image_center_opt_input,
                  _d_nearby_point_grid );
        }
#else
        for( int i=0; i<_num_cut_struct_grid; i++ ) {
            ImageCenter& v = _h_image_center_opt_input[i];
            if( not v._valid ) continue;
            if( v._iterations <= 0 ) continue;

            /* We search for the minimum of gridNSample x gridNSample
             * nearby points. Default for gridNSample is 5.
             * It is therefore most efficient to use a single-warp kernel
             * for the search.
             */
            const int gridSquare = gridNSample * gridNSample;

            if( gridSquare < 32 ) {
                popart::identification::idBestNearbyPoint31max
                    <<<1,32,0,_tag_stream>>>
                    ( getNearbyPointGridBuffer( v._tagIndex ), gridNSample );
            } else {
                cerr << __FILE__ << ":" << __LINE__
                     << " Untested code idBestNearbyPoint32plus" << endl;
                popart::identification::idBestNearbyPoint32plus
                    <<<1,32,0,_tag_stream>>>
                    ( getNearbyPointGridBuffer( v._tagIndex ), gridNSample );
            }
#endif

            v._iterations -= 1;
        }
#endif

        currentNeighbourSize /= (float)((gridNSample-1)/2) ;

        first_iteration = false;
    }
}

__host__
void TagPipe::imageCenterOptLoop( )
{
    idCostFunction( );

    for( int i=0; i<_num_cut_struct_grid; i++ ) {
        const ImageCenter& v = _h_image_center_opt_input[i];

        if( v._valid ) {
            /* When this kernel finishes, the best point does not
            * exist or it is stored in point_buffer[0]
            */
            NearbyPointGrid*   d_nearbyPointGrid = getNearbyPointGridBuffer( v._tagIndex );
            const NearbyPoint* dev_ptr           = &d_nearbyPointGrid->getGrid(0,0);

            /* This copy operation is initiated in imageCenterOptLoop instead
             * if imageCenterRetrieve (where it is needed) because the async
             * copy can run in the background.
             *
             * A SYNC IS NEEDED
             */
            POP_CUDA_MEMCPY_TO_HOST_ASYNC( v._cctag_pointer_buffer,
                                           dev_ptr,
                                           sizeof(popart::NearbyPoint),
                                           _tag_stream );
            POP_CHK_CALL_IFSYNC;
        } else {
            /* bogus values */
            v._cctag_pointer_buffer->point = make_float2( 0, 0 );
            v._cctag_pointer_buffer->result = 0.0001f;
            v._cctag_pointer_buffer->resSize = 0;
            v._cctag_pointer_buffer->readable = false;
            v._cctag_pointer_buffer->residual = 1000.0f;
        }
    }
}

__host__
bool TagPipe::imageCenterRetrieve(
    const int                           tagIndex,     // in - determines index in cut structure
    float2&                             bestPointOut, // out
    float&                              bestResidual, // out
    popart::geometry::matrix3x3&        bestHomographyOut, // out
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    if( not cctag_pointer_buffer->readable ) {
        return false;
    }

    bestPointOut      = cctag_pointer_buffer->point;
    bestHomographyOut = cctag_pointer_buffer->mHomography;
    bestResidual      = cctag_pointer_buffer->residual;
    return true;
}

__host__
void TagPipe::reallocCutStructGridBuffer( int numTags )
{
    if( numTags <= _num_cut_struct_grid ) return;

    if( _num_cut_struct_grid != 0 ) {
        POP_CUDA_FREE( _d_cut_struct_grid );
        POP_CUDA_FREE_HOST( _h_cut_struct_grid );
        POP_CUDA_FREE( _d_image_center_opt_input );
        POP_CUDA_FREE_HOST( _h_image_center_opt_input );
    }

    void* ptr;

    POP_CUDA_MALLOC( &ptr, numTags*sizeof(CutStructGrid) );
    _d_cut_struct_grid = (CutStructGrid*)ptr;

    POP_CUDA_MALLOC_HOST( &ptr, numTags*sizeof(CutStructGrid) );
    _h_cut_struct_grid = (CutStructGrid*)ptr;

    POP_CUDA_MALLOC( &ptr, numTags*sizeof(ImageCenter) );
    _d_image_center_opt_input = (ImageCenter*)ptr;

    POP_CUDA_MALLOC_HOST( &ptr, numTags*sizeof(ImageCenter) );
    _h_image_center_opt_input = (ImageCenter*)ptr;

    for( int i=0; i<_num_cut_struct_grid; i++ ) {
        _h_image_center_opt_input[i].setInvalid();
    }

    _num_cut_struct_grid = numTags;
}

__host__
void TagPipe::reallocNearbyPointGridBuffer( int numTags )
{
    if( numTags <= _num_nearby_point_grid ) return;

    if( _num_nearby_point_grid != 0 ) {
        POP_CUDA_FREE( _d_nearby_point_grid );
    }

    void* ptr;

    POP_CUDA_MALLOC( &ptr, numTags*sizeof(NearbyPointGrid) );
    _d_nearby_point_grid   = (NearbyPointGrid*)ptr;
    _num_nearby_point_grid = numTags;
}

__host__
void TagPipe::reallocSignalGridBuffer( int numTags )
{
    if( numTags <= _num_cut_signal_grid ) return;

    if( _num_cut_signal_grid != 0 ) {
        POP_CUDA_FREE( _d_cut_signal_grid );

    }
    void* ptr;

    POP_CUDA_MALLOC( &ptr, numTags*sizeof(CutSignalGrid) );
    _d_cut_signal_grid   = (CutSignalGrid*)ptr;
    _num_cut_signal_grid = numTags;
}

__host__
void TagPipe::freeCutStructGridBuffer( )
{
    if( _num_cut_struct_grid == 0 ) return;

    POP_CUDA_FREE( _d_cut_struct_grid );
    POP_CUDA_FREE_HOST( _h_cut_struct_grid );
    POP_CUDA_FREE( _d_image_center_opt_input );
    POP_CUDA_FREE_HOST( _h_image_center_opt_input );
    _num_cut_struct_grid = 0;
}

__host__
void TagPipe::freeNearbyPointGridBuffer( )
{
    if( _num_nearby_point_grid == 0 ) return;

    POP_CUDA_FREE( _d_nearby_point_grid );
    _num_nearby_point_grid = 0;
}

__host__
void TagPipe::freeSignalGridBuffer( )
{
    if( _num_cut_signal_grid == 0 ) return;

    POP_CUDA_FREE( _d_cut_signal_grid );
    _num_cut_signal_grid = 0;
}

__host__
CutStructGrid* TagPipe::getCutStructGridBufferDev( int tagIndex ) const
{
    if( tagIndex < 0 || tagIndex >= _num_cut_signal_grid ) {
        cerr << __FILE__ << ":" << __LINE__ << " ERROR: accessing a nearby point grid out of bounds" << endl;
        exit( -1 );
    }
    return &_d_cut_struct_grid[tagIndex];
}

__host__
CutStructGrid* TagPipe::getCutStructGridBufferHost( int tagIndex ) const
{
    if( tagIndex < 0 || tagIndex >= _num_cut_signal_grid ) {
        cerr << __FILE__ << ":" << __LINE__ << " ERROR: accessing a nearby point grid out of bounds" << endl;
        exit( -1 );
    }
    return &_h_cut_struct_grid[tagIndex];
}

__host__
NearbyPointGrid* TagPipe::getNearbyPointGridBuffer( int tagIndex ) const
{
    if( tagIndex < 0 || tagIndex >= _num_nearby_point_grid ) {
        cerr << __FILE__ << ":" << __LINE__ << " ERROR: accessing a nearby point grid out of bounds" << endl;
        exit( -1 );
    }
    return &_d_nearby_point_grid[tagIndex];
}

__host__
CutSignalGrid* TagPipe::getSignalGridBuffer( int tagIndex ) const
{
    if( tagIndex < 0 || tagIndex >= _num_nearby_point_grid ) {
        cerr << __FILE__ << ":" << __LINE__ << " ERROR: accessing a nearby point grid out of bounds" << endl;
        exit( -1 );
    }
    return &_d_cut_signal_grid[tagIndex];
}

}; // namespace popart

