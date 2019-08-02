/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/cctag_cuda_runtime.h>

#include "tag.h"
#include "frame.h"
#include "frameparam.h"
#include "clamp.h"
#include "assist.h"
#include "geom_matrix.h"
#include "nearby_point.h"
#include "tag_cut.h"

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

namespace cctag {

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
                                   const cctag::geometry::matrix3x3& mHomography,
                                   const cctag::geometry::matrix3x3& mInvHomography )
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

        if( cctag::any( breaknow ) )
        {
            if( threadIdx.x == 0 ) signals.outOfBounds = 1;
            return;
        }

        // Bilinear interpolation
        signals.sig[i] = cctag::identification::getPixelBilinear( src, xyRes );

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
    const cctag::geometry::ellipse    ellipse,
    const cctag::geometry::matrix3x3  mT,
    float2                             center,
    const float                        neighbourSize,
    NearbyPointGrid*                   d_nearbyPointGrid )
{
    const size_t gridNSample = STRICT_SAMPLE(tagParam.gridNSample);

    assert( gridDim.y == STRICT_SAMPLE(gridNSample) );
    assert( gridDim.z == STRICT_SAMPLE(gridNSample) );

    const float  gridWidth   = neighbourSize;
    const float  halfWidth   = gridWidth/2.0f;
    const float  stepSize    = gridWidth * __frcp_rn( float(STRICT_SAMPLE(gridNSample)-1) );

    if( ! first_iteration ) {
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

    cctag::geometry::matrix3x3  mInvT;
    mT.invert( mInvT ); // note: returns false if it fails
    mInvT.condition( condCenter );

    nPoint.point    = condCenter;
    nPoint.result   = 0.0f;
    nPoint.resSize  = 0;
    nPoint.readable = true;
    ellipse.computeHomographyFromImagedCenter( nPoint.point, nPoint.mHomography );
    nPoint.mHomography.invert( nPoint.mInvHomography );
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
                  ! l_signals.outOfBounds &&
                  ! r_signals.outOfBounds;
        if( comp ) {
            const int limit = STRICT_SIGSIZE(cut_buffer->getGrid(left_cut).sigSize); // we could also use right_cut
            for( int offset = threadIdx.x; offset < STRICT_SIGSIZE(limit); offset += 32 ) {
                float square = l_signals.sig[offset] - r_signals.sig[offset];
                val += ( square * square );
            }
            ct = 1;
        }
    }

    val += cctag::shuffle_down( val, 16 );
    val += cctag::shuffle_down( val,  8 );
    val += cctag::shuffle_down( val,  4 );
    val += cctag::shuffle_down( val,  2 );
    val += cctag::shuffle_down( val,  1 );

    __shared__ float signal_sum[32];
    __shared__ int   count_sum[32];

    if( threadIdx.x == 0 ) {
        signal_sum[threadIdx.y] = val;
        count_sum [threadIdx.y] = ct;
    }

    __syncthreads();

    if( threadIdx.y == 0 ) {
        val = signal_sum[threadIdx.x];
        val += cctag::shuffle_down( val, 16 );
        val += cctag::shuffle_down( val,  8 );
        val += cctag::shuffle_down( val,  4 );
        val += cctag::shuffle_down( val,  2 );
        val += cctag::shuffle_down( val,  1 );
        ct  = count_sum[threadIdx.x];
        ct  += cctag::shuffle_down( ct, 16 );
        ct  += cctag::shuffle_down( ct,  8 );
        ct  += cctag::shuffle_down( ct,  4 );
        ct  += cctag::shuffle_down( ct,  2 );
        ct  += cctag::shuffle_down( ct,  1 );

        if( threadIdx.x == 0 ) {
            atomicAdd( &nPoint.result,  val );
            atomicAdd( &nPoint.resSize, ct );
        }
    }
}

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
        int otherRes = cctag::shuffle_down( bestRes, (1 << shft) );
        int otherIdx = cctag::shuffle_down( bestIdx, (1 << shft) );
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
        int otherRes = cctag::shuffle_down( bestRes, (1 << shft) );
        int otherIdx = cctag::shuffle_down( bestIdx, (1 << shft) );
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
bool TagPipe::idCostFunction(
    const int                           tagIndex,
    const int                           debug_numTags,
    cudaStream_t                        tagStream,
    int                                 iterations,
    const cctag::geometry::ellipse&    ellipse,
    const float2                        center,
    const int                           vCutSize,
    float                               currentNeighbourSize,
    const cctag::Parameters&            params )
{
    if( vCutSize < 2 ) return false;

    const size_t gridNSample = params._imagedCenterNGridSample;
    const size_t offset      = tagIndex * STRICT_SAMPLE(gridNSample) * STRICT_SAMPLE(gridNSample);

    /* reusing various image-sized plane */
    NearbyPointGrid* d_NearbyPointGrid  = getNearbyPointGridBuffer( tagIndex );

    const CutStructGrid* cut_buffer = getCutStructGridBufferDev( tagIndex );
    CutSignalGrid*       sig_buffer = getSignalGridBuffer( tagIndex );

    cctag::geometry::matrix3x3 mT;
    ellipse.makeConditionerFromEllipse( mT );
    cctag::geometry::matrix3x3 mInvT;
    bool success;
    success = mT.invert( mInvT ); // note: returns false if it fails
    if( ! success ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Conditioner matrix extracted from ellipse is not invertable" << endl
             << "    Program logic error. Requires analysis before fixing." << endl
             << endl;
    }

    cctag::geometry::ellipse transformedEllipse;
    ellipse.projectiveTransform( mInvT, transformedEllipse );

    bool first_iteration = true;

    for( ; iterations>0; iterations-- ) {
        float neighSize = currentNeighbourSize * max( transformedEllipse.a(),
                                                      transformedEllipse.b() );

        dim3 block( 1, 1, 1 );
        dim3 grid( 1, STRICT_SAMPLE(gridNSample), STRICT_SAMPLE(gridNSample) );

        cctag::identification::initAllNearbyPoints
            <<<grid,block,0,tagStream>>>
            ( first_iteration,
              ellipse,
              mT,
              center,
              neighSize,
              d_NearbyPointGrid );

        dim3 get_block( 32, STRICT_CUTSIZE(vCutSize), 1 ); // we use this to sum up signals
        dim3 get_grid( 1, STRICT_SAMPLE(gridNSample), STRICT_SAMPLE(gridNSample) );

        cctag::identification::idGetSignals
            <<<get_grid,get_block,0,tagStream>>>
            ( _frame[0]->getPlaneDev(),
              STRICT_CUTSIZE(vCutSize),
              d_NearbyPointGrid,        // in
              cut_buffer,
              sig_buffer );

        dim3 id_block( 32, // we use this to sum up signals
                       32, // we can use some shared memory/warp magic for summing
                       1 );
        const int numPairs = STRICT_CUTSIZE(vCutSize)*(STRICT_CUTSIZE(vCutSize)-1)/2;
        dim3 id_grid( grid_divide( numPairs, 32 ),
                      STRICT_SAMPLE(gridNSample),
                      STRICT_SAMPLE(gridNSample) );

        cctag::identification::idComputeResult
            <<<id_grid,id_block,0,tagStream>>>
            ( d_NearbyPointGrid,
              cut_buffer,
              sig_buffer,
              STRICT_CUTSIZE(vCutSize) );

        /* We search for the minimum of gridNSample x gridNSample
         * nearby points. Default for gridNSample is 5.
         * It is therefore most efficient to use a single-warp kernel
         * for the search.
         */
        const int gridSquare = STRICT_SAMPLE(gridNSample) * STRICT_SAMPLE(gridNSample);

        if( gridSquare < 32 ) {
            cctag::identification::idBestNearbyPoint31max
                <<<1,32,0,tagStream>>>
                  ( d_NearbyPointGrid, STRICT_SAMPLE(gridNSample) );
        } else {
cerr << __FILE__ << ":" << __LINE__ << " Untested code idBestNearbyPoint32plus" << endl;
            cctag::identification::idBestNearbyPoint32plus
                <<<1,32,0,tagStream>>>
                  ( d_NearbyPointGrid, STRICT_SAMPLE(gridNSample) );
        }

        currentNeighbourSize /= (float)((STRICT_SAMPLE(gridNSample)-1)/2) ;

        first_iteration = false;
    }

    return true;
}

__host__
void TagPipe::imageCenterOptLoop(
    const int                           tagIndex,     // in - determines index in cut structure
    const int                           debug_numTags, // in - only for debugging
    cudaStream_t                        tagStream,
    const cctag::geometry::ellipse&    outerEllipse, // in
    const float2&                       center,       // in
    const int                           vCutSize,
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    if( vCutSize != 22 ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    " << __func__ << " is called from CPU code with vCutSize " << vCutSize << " instead of 22" << endl;
        if( vCutSize > 22 ) {
            exit( -1 );
        }
    }

    const float  maxSemiAxis   = std::max( outerEllipse.a(), outerEllipse.b() );
    const size_t gridNSample   = params._imagedCenterNGridSample;
    float        neighbourSize = params._imagedCenterNeighbourSize;
    int          iterations    = 0;

    while( neighbourSize*maxSemiAxis > 0.02 ) {
        iterations += 1;
        neighbourSize /= (float)((STRICT_SAMPLE(gridNSample)-1)/2) ;
    }

    neighbourSize = params._imagedCenterNeighbourSize;

    NearbyPointGrid* d_nearbyPointGrid = getNearbyPointGridBuffer( tagIndex );

    bool success = idCostFunction( tagIndex,
                                   debug_numTags,
                                   tagStream,
                                   iterations,
                                   outerEllipse,
                                   center,
                                   STRICT_CUTSIZE(vCutSize),
                                   neighbourSize,
                                   params );

    if( success ) {
        /* When this kernel finishes, the best point does not
         * exist or it is stored in point_buffer[0]
         */
        const NearbyPoint* dev_ptr = &d_nearbyPointGrid->getGrid(0,0);

        /* This copy operation is initiated in imageCenterOptLoop instead
         * if imageCenterRetrieve (where it is needed) because the async
         * copy can run in the background.
         *
         * A SYNC IS NEEDED
         */
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( cctag_pointer_buffer,
                                       dev_ptr,
                                       sizeof(cctag::NearbyPoint),
                                       tagStream );
        POP_CHK_CALL_IFSYNC;
    } else {
        /* bogus values */
        cctag_pointer_buffer->point = make_float2( 0, 0 );
        cctag_pointer_buffer->result = 0.0001f;
        cctag_pointer_buffer->resSize = 0;
        cctag_pointer_buffer->readable = false;
        cctag_pointer_buffer->residual = 1000.0f;
    }
}

__host__
bool TagPipe::imageCenterRetrieve(
    const int                           tagIndex,     // in - determines index in cut structure
    cudaStream_t                        tagStream,
    float2&                             bestPointOut, // out
    float&                              bestResidual, // out
    cctag::geometry::matrix3x3&        bestHomographyOut, // out
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    if( !cctag_pointer_buffer || !cctag_pointer_buffer->readable ) {
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
    }

    void* ptr;

    POP_CUDA_MALLOC( &ptr, numTags*sizeof(CutStructGrid) );
    _d_cut_struct_grid = (CutStructGrid*)ptr;

    POP_CUDA_MALLOC_HOST( &ptr, numTags*sizeof(CutStructGrid) );
    _h_cut_struct_grid = (CutStructGrid*)ptr;

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

}; // namespace cctag

