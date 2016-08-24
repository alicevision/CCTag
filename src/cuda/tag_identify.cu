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
    const popart::geometry::ellipse    ellipse,
    const popart::geometry::matrix3x3  mT,
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

} // namespace identification

/**
 * @pre the cuts for this tag have been uploaded.
 * @param[in] i the index in the ImageCenter vector
 * @param[in] iterations the caller defines how many refinement loop we execute
 */
__host__
void TagPipe::idCostFunction( vector<bool>& success )
{
    const size_t gridNSample = _params._imagedCenterNGridSample;

    for( int i=0; i<_num_cut_struct_grid; i++ ) {
        ImageCenter& v = _h_image_center_opt_input[i];

        if( not v._valid ) {
            success[i] = false;
            continue;
        }

        cudaStream_t& tagStream = _tag_streams[ v._tagIndex % NUM_ID_STREAMS ];

        const CutStructGrid* cut_buffer = getCutStructGridBufferDev( v._tagIndex );

        float currentNeighbourSize = _params._imagedCenterNeighbourSize;

        bool first_iteration = true;

        for( ; v._iterations>0; v._iterations-- ) {
            float neighSize = currentNeighbourSize * v._transformedEllipseMaxRadius;

            dim3 block( 1, 1, 1 );
            dim3 grid( 1, STRICT_SAMPLE(gridNSample), STRICT_SAMPLE(gridNSample) );

            popart::identification::initAllNearbyPoints
                <<<grid,block,0,tagStream>>>
                ( first_iteration,
                  v._outerEllipse,
                  v._mT,
                  v._center,
                  neighSize,
                  getNearbyPointGridBuffer( v._tagIndex ) );

            dim3 get_block( 32, STRICT_CUTSIZE(v._vCutSize), 1 ); // we use this to sum up signals
            dim3 get_grid( 1, STRICT_SAMPLE(gridNSample), STRICT_SAMPLE(gridNSample) );

            popart::identification::idGetSignals
                <<<get_grid,get_block,0,tagStream>>>
                ( _frame[0]->getPlaneDev(),
                  STRICT_CUTSIZE(v._vCutSize),
                  getNearbyPointGridBuffer( v._tagIndex ),        // in
                  cut_buffer,
                  getSignalGridBuffer( v._tagIndex ) );

            dim3 id_block( 32, // we use this to sum up signals
                        32, // we can use some shared memory/warp magic for summing
                        1 );
            const int numPairs = STRICT_CUTSIZE(v._vCutSize)*(STRICT_CUTSIZE(v._vCutSize)-1)/2;
            dim3 id_grid( grid_divide( numPairs, 32 ),
                          STRICT_SAMPLE(gridNSample),
                          STRICT_SAMPLE(gridNSample) );

            popart::identification::idComputeResult
                <<<id_grid,id_block,0,tagStream>>>
                ( getNearbyPointGridBuffer( v._tagIndex ),
                  cut_buffer,
                  getSignalGridBuffer( v._tagIndex ),
                  STRICT_CUTSIZE(v._vCutSize) );

            /* We search for the minimum of gridNSample x gridNSample
            * nearby points. Default for gridNSample is 5.
            * It is therefore most efficient to use a single-warp kernel
            * for the search.
            */
            const int gridSquare = STRICT_SAMPLE(gridNSample) * STRICT_SAMPLE(gridNSample);

            if( gridSquare < 32 ) {
                popart::identification::idBestNearbyPoint31max
                    <<<1,32,0,tagStream>>>
                    ( getNearbyPointGridBuffer( v._tagIndex ), STRICT_SAMPLE(gridNSample) );
            } else {
                cerr << __FILE__ << ":" << __LINE__ << " Untested code idBestNearbyPoint32plus" << endl;
                popart::identification::idBestNearbyPoint32plus
                    <<<1,32,0,tagStream>>>
                    ( getNearbyPointGridBuffer( v._tagIndex ), STRICT_SAMPLE(gridNSample) );
            }

            currentNeighbourSize /= (float)((STRICT_SAMPLE(gridNSample)-1)/2) ;

            first_iteration = false;
        }

        success[i] = true;
    }
}

__host__
void TagPipe::imageCenterOptLoop( )
{
    vector<bool> success( _num_cut_struct_grid );

    idCostFunction( success );

    for( int i=0; i<_num_cut_struct_grid; i++ ) {
        const ImageCenter& v = _h_image_center_opt_input[i];

        if( not v._valid ) continue;

        if( success[i] ) {
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
            cudaStream_t& tagStream = _tag_streams[ v._tagIndex % NUM_ID_STREAMS ];

            POP_CUDA_MEMCPY_TO_HOST_ASYNC( v._cctag_pointer_buffer,
                                           dev_ptr,
                                           sizeof(popart::NearbyPoint),
                                           tagStream );
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
        _h_image_center_opt_input[i]->setInvalid();
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

