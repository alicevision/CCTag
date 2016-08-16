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

#undef NO_ATOMIC

using namespace std;

namespace popart {

namespace identification {

__device__
inline
float getPixelBilinear( cv::cuda::PtrStepSzb src, float2 xy )
{
    const int px = (int)xy.x; // floor of x
    const int py = (int)xy.y; // floor of y

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
                                   CutSignals*                        signals,
                                   cv::cuda::PtrStepSzb               src,
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
    const std::size_t nSamples = cut.sigSize;
    const float stepX = ( xStop - xStart ) / ( nSamples - 1.0f );
    const float stepY = ( yStop - yStart ) / ( nSamples - 1.0f );
    const float stepX32 = 32.0f * stepX;
    const float stepY32 = 32.0f * stepY;

    // float x =  xStart; - serial code
    // float y =  yStart; - serial code
    float x =  xStart + threadIdx.x * stepX;
    float y =  yStart + threadIdx.x * stepY;
    for( std::size_t i = threadIdx.x; i < nSamples; i += 32 ) {
        float2 xyRes;

        // [xRes;yRes;1] ~= mHomography*[x;y;1.0]
        xyRes = mHomography.applyHomography( x, y );

        bool breaknow = ( xyRes.x < 1.0f && xyRes.x > src.cols-1 && xyRes.y < 1.0f && xyRes.y > src.rows-1 );

        if( __any( breaknow ) )
        {
            if( threadIdx.x == 0 ) signals->outOfBounds = 1;
            return;
        }

        // Bilinear interpolation
        signals->sig[i] = popart::identification::getPixelBilinear( src, xyRes );

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
void idGetSignals( cv::cuda::PtrStepSzb src,
                   const int            vCutSize,
                   const NearbyPoint*   point_buffer,
                   const CutStruct*     cut_buffer,
                   CutSignals*          sig_buffer )
{
    const size_t gridNSample = tagParam.gridNSample;
    const int i = blockIdx.y;
    const int j = blockIdx.z;
    const int idx = j * gridNSample + i;

    const NearbyPoint* nPoint = &point_buffer[idx];
    identification::CutSignals* signals = &sig_buffer[idx * vCutSize];

    int myCutIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if( myCutIdx >= vCutSize ) {
        // warps do never loose lockstep because of this
        return; // out of bounds
    }

    const CutStruct& myCut     = cut_buffer[myCutIdx];
    CutSignals*      mySignals = &signals[myCutIdx];

    if( threadIdx.x == 0 ) mySignals->outOfBounds = 0;

    extractSignalUsingHomography( myCut,
                                  mySignals,
                                  src,
                                  nPoint->mHomography,
                                  nPoint->mInvHomography );
}

__global__
void initAllNearbyPoints(
    bool                               first_iteration,
    const popart::geometry::ellipse    ellipse,
    const popart::geometry::matrix3x3  mT,
    float2                             center,
    const float                        neighbourSize,
    NearbyPoint*                       point_buffer )
{
    const size_t gridNSample = tagParam.gridNSample;

    assert( gridDim.y == gridNSample );
    assert( gridDim.z == gridNSample );

    const float  gridWidth   = neighbourSize;
    const float  halfWidth   = gridWidth/2.0f;
    const float  stepSize    = gridWidth * __frcp_rn( float(gridNSample-1) );

    if( not first_iteration ) {
        // the best center is located in point_buffer[0]
        center = point_buffer[0].point;
        // @lilian: why do we need this "center = mT * center" in every iteration?
    }
    mT.condition( center );

    const int i = blockIdx.y;
    const int j = blockIdx.z;
    const int idx = j * gridNSample + i;


    NearbyPoint* nPoint = &point_buffer[idx];
    // identification::CutSignals* signals = &sig_buffer[idx * vCutSize];

    float2 condCenter = make_float2( center.x - halfWidth + i*stepSize,
                                     center.y - halfWidth + j*stepSize );

    popart::geometry::matrix3x3  mInvT;
    mT.invert( mInvT ); // note: returns false if it fails
    mInvT.condition( condCenter );

    nPoint->point    = condCenter;
    nPoint->result   = 0.0f;
    nPoint->resSize  = 0;
    nPoint->readable = true;
    ellipse.computeHomographyFromImagedCenter( nPoint->point, nPoint->mHomography );
    nPoint->mHomography.invert( nPoint->mInvHomography );
}

__global__
void verify_idComputeResult( NearbyPoint*      point_buffer,
                             const CutStruct*  cut_buffer,
                             const CutSignals* sig_buffer,
                             const int         vCutSize,
                             const int         numPairs )
{
    const size_t gridNSample = tagParam.gridNSample; // this is a global constant !
    // const int grid_i   = blockIdx.y; - an X mesh point 0..gridNSample-1
    // const int grid_j   = blockIdx.z; - a  Y mesh point 0..gridNSample-1
    for( int grid_i=0; grid_i<gridNSample; grid_i++ ) {
        for( int grid_j=0; grid_j<gridNSample; grid_j++ ) {
            const int grid_idx = grid_j * gridNSample + grid_i;
            const CutSignals* allcut_signals = &sig_buffer[grid_idx * vCutSize];
            // NearbyPoint* const nPoint  = &point_buffer[grid_idx];

            const int blockIdx_x_limit = grid_divide( numPairs, 32 );
            // for( int blockIdx_x=0; blockIdx_x<blockIdx_x_limit; blockIdx_x++ ) {
            int blockIdx_x=blockIdx_x_limit-1; {
                const int threadIdx_y_limit = 32;
                const int threadIdx_x_limit = 32;
                for( int threadIdx_y=0; threadIdx_y<threadIdx_y_limit; threadIdx_y++ ) {
                  for( int threadIdx_x=0; threadIdx_x<threadIdx_x_limit; threadIdx_x++ ) {
                    int myPair = blockIdx_x * 32 + threadIdx_y;
                    int j      = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
                    int i      = myPair - j*(j-1)/2;

                    float val  = 0.0f;
                    bool  comp = true;

                    comp = ( j < vCutSize && i < j );

                    if( comp ) {
                        const CutSignals* l_signals = &allcut_signals[i];
                        const CutSignals* r_signals = &allcut_signals[j];
                        // default sample cut length is 100
                        comp  = ( threadIdx_x < tagParam.sampleCutLength ) &&
                                not l_signals->outOfBounds &&
                                not r_signals->outOfBounds;
                        if( comp ) {
                            // sigSize is always 100
                            // anything else is an indication of major trouble
                            if( cut_buffer[i].sigSize != 100 ) {
                                printf("We read a sigSize that is not 100\n");
                                printf("grid_i: %d grid_j: %d gridNSample: %d\n", grid_i, grid_j, gridNSample);
                                printf("blockIdx_x: %d threadIdx_x: %d threadIdx_y: %d\n", blockIdx_x, threadIdx_x, threadIdx_y );
                                printf("vCutSize: %d numPairs: %d\n", vCutSize, numPairs );
                                printf("myPair: %d i: %d j: %d\n", myPair, i, j );
                                return;
                            }
                            const int limit = cut_buffer[i].sigSize;
                            for( int offset = threadIdx_x; offset < limit; offset += 32 ) {
                                float square = l_signals->sig[offset] - r_signals->sig[offset];
                                val += ( square * square );
                            }
                        }
                    }
                  }
                }
            }
        }
    }
}

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
void idComputeResult( NearbyPoint*      point_buffer,
                      const CutStruct*  cut_buffer,
                      const CutSignals* sig_buffer,
                      const int         vCutSize )
{
  const size_t gridNSample = tagParam.gridNSample;
  const int grid_i   = blockIdx.y;
  const int grid_j   = blockIdx.z;
  const int grid_idx = grid_j * gridNSample + grid_i;
  const CutSignals* allcut_signals = &sig_buffer[grid_idx * vCutSize];
  NearbyPoint* const nPoint  = &point_buffer[grid_idx];

#ifdef NO_ATOMIC
  const int numPairs = vCutSize*(vCutSize-1)/2;
  for( int myPair = threadIdx.y; myPair < numPairs; myPair += 32 ) {
#else
    int myPair = blockIdx.x * 32 + threadIdx.y;
#endif
    int j      = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
    int i      = myPair - j*(j-1)/2;

    int   ct   = 0;
    float val  = 0.0f;
    bool  comp = true;

    comp = ( j < vCutSize && i < j );

    if( comp ) {
        const CutSignals* l_signals = &allcut_signals[i];
        const CutSignals* r_signals = &allcut_signals[j];
        comp  = ( threadIdx.x < tagParam.sampleCutLength ) &&
                  not l_signals->outOfBounds &&
                  not r_signals->outOfBounds;
        if( comp ) {
            const int limit = cut_buffer[i].sigSize; // we could also use j ?
            for( int offset = threadIdx.x; offset < limit; offset += 32 ) {
                float square = l_signals->sig[offset] - r_signals->sig[offset];
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

#if 0
    if( threadIdx.x == 0 ) {
        printf("idComputeResult (j,i)=(%d,%d) val=%f ct=%d\n",j,i,val,ct);
    }
#endif

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
#ifdef NO_ATOMIC
            nPoint->result  += val;
            nPoint->resSize += ct;
#else
            atomicAdd( &nPoint->result,  val );
            atomicAdd( &nPoint->resSize, ct );
#endif
        }
    }
#ifdef NO_ATOMIC
    __syncthreads();
  }
#endif
}

__global__
void idBestNearbyPoint32plus( NearbyPoint* point_buffer, const size_t gridSquare )
{
    // phase 1: each thread searches for its own best point
    float bestRes = FLT_MAX;
    int   bestIdx = gridSquare-1;
    int   idx;
    for( idx=threadIdx.x; idx<gridSquare; idx+=32 ) {
        const NearbyPoint& point = point_buffer[idx];
        if( point.readable ) {
            bestIdx = idx;
            bestRes = point.result / point.resSize;
            break;
        }
    }
    __syncthreads();
    for( ; idx<gridSquare; idx+=32 ) {
        const NearbyPoint& point = point_buffer[idx];
        if( point.readable ) {
            float val = point.result / point.resSize;
            if( val < bestRes ) {
                bestRes = val;
                bestIdx = idx;
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
            NearbyPoint*       dst_point = &point_buffer[0];
            const NearbyPoint* src_point = &point_buffer[bestIdx];
            memcpy( dst_point, src_point, sizeof( NearbyPoint ) );
            dst_point->residual = bestRes;
        }
    }
}

__global__
void idBestNearbyPoint31max( NearbyPoint* point_buffer, const size_t gridSquare )
{
    // phase 1: each thread retrieves its point
    float bestRes = FLT_MAX;
    int   bestIdx = gridSquare-1;
    int   idx     = threadIdx.x;
    if( idx < gridSquare ) {
        const NearbyPoint& point = point_buffer[idx];
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
            NearbyPoint*       dst_point = &point_buffer[0];
            const NearbyPoint* src_point = &point_buffer[bestIdx];
            memcpy( dst_point, src_point, sizeof( NearbyPoint ) );
            dst_point->residual = bestRes;
        }
    }
}

} // namespace identification

/**
 * @pre the cuts for this tag have been uploaded.
 * @param[in] tagIndex counter for tag, determines index in cut structure
 * @param[in] iterations the caller defines how many refinement loop we execute
 * @param[in] ellipse the parametrized formulation of the outer(?) ellipse
 * @param[in] center the approximate center of the tag
 * @param[in] vCutSize ?
 8 @param[in] currentNeighbourSize the search area for the center, decreasing with iterations
 * @param[out] bestPointOut the center of the tag after all iterations
 * @param[out] bestHomographyOut homography that rectifies the best center
 * @param[in] params gives access to various program parameters
 * @param[inout] pointer lock a small piece of page-locked memory for device-to-host copying of results
 */
__host__
void TagPipe::idCostFunction(
    const int                           tagIndex,
    cudaStream_t                        tagStream,
    int                                 iterations,
    const popart::geometry::ellipse&    ellipse,
    const float2                        center,
    const int                           vCutSize,
    float                               currentNeighbourSize,
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    const size_t gridNSample = params._imagedCenterNGridSample;
    const size_t offset      = tagIndex * gridNSample * gridNSample;

    /* reusing various image-sized plane */
    NearbyPoint*                 point_buffer;
    identification::CutSignals*  sig_buffer;
    const identification::CutStruct* cut_buffer = getCutStructBufferDev();

    point_buffer  = getNearbyPointBuffer();
    sig_buffer    = getSignalBuffer();

    point_buffer  = &point_buffer[offset];
    sig_buffer    = &sig_buffer[offset * params._numCutsInIdentStep];
    cut_buffer    = &cut_buffer[tagIndex * params._numCutsInIdentStep];

    popart::geometry::matrix3x3 mT;
    ellipse.makeConditionerFromEllipse( mT );
    popart::geometry::matrix3x3 mInvT;
    bool success = mT.invert( mInvT ); // note: returns false if it fails
    if( not success ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Conditioner matrix extracted from ellipse is not invertable" << endl
             << "    Program logic error. Requires analysis before fixing." << endl
             << endl;
    }

    popart::geometry::ellipse transformedEllipse;
    ellipse.projectiveTransform( mInvT, transformedEllipse );

    bool first_iteration = true;

    for( ; iterations>0; iterations-- ) {
        float neighSize = currentNeighbourSize * max( transformedEllipse.a(),
                                                      transformedEllipse.b() );

        dim3 block( 1, 1, 1 );
        dim3 grid( 1, gridNSample, gridNSample );

        popart::identification::initAllNearbyPoints
            <<<grid,block,0,tagStream>>>
            ( first_iteration,
              ellipse,
              mT,
              center,
              neighSize,
              point_buffer );
POP_SYNC_CHK;
        POP_CHK_CALL_IFSYNC;
        dim3 get_block( 32, vCutSize, 1 ); // we use this to sum up signals
        dim3 get_grid( 1, gridNSample, gridNSample );

        popart::identification::idGetSignals
            <<<get_grid,get_block,0,tagStream>>>
            ( _frame[0]->getPlaneDev(),
              vCutSize,
              point_buffer,
              cut_buffer,
              sig_buffer );
POP_SYNC_CHK;
        POP_CHK_CALL_IFSYNC;

        {
            const int numPairs = vCutSize*(vCutSize-1)/2;

            popart::identification::verify_idComputeResult
                <<<1,1,0,tagStream>>>
                ( point_buffer, cut_buffer, sig_buffer, vCutSize, numPairs );
POP_SYNC_CHK;
        }

        dim3 id_block( 32, // we use this to sum up signals
                       32, // we can use some shared memory/warp magic for summing
                       1 );
#ifdef NO_ATOMIC
#error no atomic
        dim3 id_grid( 1,
                      gridNSample,
                      gridNSample );
#else
        const int numPairs = vCutSize*(vCutSize-1)/2;
        dim3 id_grid( grid_divide( numPairs, 32 ),
                      gridNSample,
                      gridNSample );
#endif

        popart::identification::idComputeResult
            <<<id_grid,id_block,0,tagStream>>>
            ( point_buffer, cut_buffer, sig_buffer, vCutSize );
POP_SYNC_CHK;
        POP_CHK_CALL_IFSYNC;

        /* We search for the minimum of gridNSample x gridNSample
         * nearby points. Default for gridNSample is 5.
         * It is therefore most efficient to use a single-warp kernel
         * for the search.
         */
        const int gridSquare = gridNSample * gridNSample;

        if( gridSquare < 32 ) {
            popart::identification::idBestNearbyPoint31max
                <<<1,32,0,tagStream>>>
                  ( point_buffer, gridSquare );
POP_SYNC_CHK;
            POP_CHK_CALL_IFSYNC;
        } else {
            popart::identification::idBestNearbyPoint32plus
                <<<1,32,0,tagStream>>>
                  ( point_buffer, gridSquare );
POP_SYNC_CHK;
            POP_CHK_CALL_IFSYNC;
        }

        currentNeighbourSize /= (float)((gridNSample-1)/2) ;

        first_iteration = false;
    }
}

__host__
void TagPipe::imageCenterOptLoop(
    const int                           tagIndex,     // in - determines index in cut structure
    cudaStream_t                        tagStream,
    const popart::geometry::ellipse&    outerEllipse, // in
    const float2&                       center,       // in
    const int                           vCutSize,
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    clearSignalBuffer( );

    const float  maxSemiAxis   = std::max( outerEllipse.a(), outerEllipse.b() );
    const size_t gridNSample   = params._imagedCenterNGridSample;
    float        neighbourSize = params._imagedCenterNeighbourSize;
    int          iterations    = 0;

    while( neighbourSize*maxSemiAxis > 0.02 ) {
        iterations += 1;
        neighbourSize /= (float)((gridNSample-1)/2) ;
    }

    neighbourSize = params._imagedCenterNeighbourSize;

    idCostFunction( tagIndex,
                    tagStream,
                    iterations,
                    outerEllipse,
                    center,
                    vCutSize,
                    neighbourSize,
                    params,
                    cctag_pointer_buffer );

    NearbyPoint* point_buffer;
    point_buffer  = getNearbyPointBuffer();
    point_buffer  = &point_buffer[tagIndex * gridNSample * gridNSample];

    /* When this kernel finishes, the best point does not
     * exist or it is stored in point_buffer[0]
     */
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( cctag_pointer_buffer,
                                   point_buffer,
                                   sizeof(popart::NearbyPoint),
                                   tagStream );
    POP_CHK_CALL_IFSYNC;
}

__host__
bool TagPipe::imageCenterRetrieve(
    const int                           tagIndex,     // in - determines index in cut structure
    cudaStream_t                        tagStream,
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
void TagPipe::allocCutStructBuffer( int n )
{
    void* ptr;

    POP_CUDA_MALLOC( &ptr, n*sizeof(identification::CutStruct) );
    _d_cut_struct = (identification::CutStruct*)ptr;

    POP_CUDA_MALLOC_HOST( &ptr, n*sizeof(identification::CutStruct) );
    _h_cut_struct = (identification::CutStruct*)ptr;

    _num_cut_struct = n;
}

__host__
void TagPipe::allocNearbyPointBuffer( int n )
{
    void* ptr;

    POP_CUDA_MALLOC( &ptr, n*sizeof(NearbyPoint) );
    _d_nearby_point = (NearbyPoint*)ptr;

    _num_nearby_point = n;
}

__host__
void TagPipe::allocSignalBuffer( int n )
{
    void* ptr;

    POP_CUDA_MALLOC( &ptr, n*sizeof(identification::CutSignals) );
    _d_cut_signals = (identification::CutSignals*)ptr;

    _num_cut_signals = n;
}

__host__
void TagPipe::freeCutStructBuffer( )
{
    POP_CUDA_FREE( _d_cut_struct );
    POP_CUDA_FREE_HOST( _h_cut_struct );
    _num_cut_struct = 0;
}

__host__
void TagPipe::freeNearbyPointBuffer( )
{
    POP_CUDA_FREE( _d_nearby_point );
    _num_nearby_point = 0;
}

__host__
void TagPipe::freeSignalBuffer( )
{
    POP_CUDA_FREE( _d_cut_signals );
    _num_cut_signals = 0;
}

__host__
size_t TagPipe::getCutStructBufferByteSize( ) const
{
    return _num_cut_struct * sizeof(identification::CutStruct);
}

__host__
identification::CutStruct* TagPipe::getCutStructBufferDev( ) const
{
    return _d_cut_struct;
}

__host__
identification::CutStruct* TagPipe::getCutStructBufferHost( ) const
{
    return _h_cut_struct;
}

__host__
size_t TagPipe::getNearbyPointBufferByteSize( ) const
{
    return _num_nearby_point * sizeof(NearbyPoint);
}

__host__
NearbyPoint* TagPipe::getNearbyPointBuffer( ) const
{
    return _d_nearby_point;
}

__host__
size_t TagPipe::getSignalBufferByteSize( ) const
{
    return _num_cut_signals * sizeof(identification::CutSignals);
}

__host__
identification::CutSignals* TagPipe::getSignalBuffer( ) const
{
    return _d_cut_signals;
}

__host__
void TagPipe::clearSignalBuffer( )
{
#ifdef DEBUG_FRAME_UPLOAD_CUTS
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_MEMSET_ASYNC( 0, // _d_intermediate.data,
                           -1,
                           0, // _h_intermediate.step * _h_intermediate.rows,
                           _stream );
    POP_CHK_CALL_IFSYNC;
#endif // DEBUG_FRAME_UPLOAD_CUTS
}

}; // namespace popart

