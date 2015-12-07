#include <cuda_runtime.h>

#include "frame.h"
#include "frameparam.h"
#include "clamp.h"
#include "geom_matrix.h"
#include "geom_projtrans.h"
#include "nearby_point.h"
#include "tag_cut.h"

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
void idGetSignals( const int            tagIndex,
                   NearbyPoint*         nPoint,
                   cv::cuda::PtrStepSzb src,
                   const CutStruct*     cuts,
                   CutSignals*          signals,
                   const int            vCutsSize )
{
    int myCutIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if( myCutIdx >= vCutsSize ) {
        // warps do never loose lockstep because of this
        return; // out of bounds
    }

    const CutStruct& myCut     = cuts[tagIndex * tagParam.numCutsInIdentStep + myCutIdx];
    CutSignals*      mySignals = &signals[myCutIdx];

    if( threadIdx.x == 0 ) mySignals->outOfBounds = 0;

    extractSignalUsingHomography( myCut,
                                  mySignals,
                                  src,
                                  nPoint->mHomography,
                                  nPoint->mInvHomography );
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
void idComputeResult( const int         tagIndex,
                      NearbyPoint*      point_buffer,
                      const CutStruct*  vCuts,
                      const CutSignals* sig_buffer,
                      const int         vCutsSize )
{
  const size_t gridNSample = tagParam.gridNSample;
  const int grid_i   = blockIdx.y;
  const int grid_j   = blockIdx.z;
  const int grid_idx = grid_j * gridNSample + grid_i;
  const CutSignals* allcut_signals = &sig_buffer[grid_idx * vCutsSize];
  NearbyPoint* const nPoint  = &point_buffer[grid_idx];

#ifdef NO_ATOMIC
  const int numPairs = vCutsSize*(vCutsSize-1)/2;
  for( int myPair = threadIdx.y; myPair < numPairs; myPair += 32 ) {
#else
    int myPair = blockIdx.x * 32 + threadIdx.y;
#endif
    int j      = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
    int i      = myPair - j*(j-1)/2;

    int   ct   = 0;
    float val  = 0.0f;
    bool  comp = true;

    comp = ( j < vCutsSize && i < j );

    if( comp ) {
        const CutSignals* l_signals = &allcut_signals[i];
        const CutSignals* r_signals = &allcut_signals[j];
        comp  = ( threadIdx.x < tagParam.sampleCutLength ) &&
                  not l_signals->outOfBounds &&
                  not r_signals->outOfBounds;
        if( comp ) {
            const int limit = vCuts[tagIndex * tagParam.numCutsInIdentStep + i].sigSize; // we could also use j ?
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
void getSignalsAllNearbyPoints(
    const int                          tagIndex,
    bool                               first_iteration,
    cv::cuda::PtrStepSzb               src,
    const popart::geometry::ellipse    ellipse,
    const popart::geometry::matrix3x3  mT,
    // const popart::geometry::matrix3x3  mInvT,
    float2                             center,
    const int                          vCutsSize,
    const float                        neighbourSize,
    NearbyPoint*                       point_buffer,
    const identification::CutStruct*   cut_buffer,
    identification::CutSignals*        sig_buffer )
{
    const size_t gridNSample = tagParam.gridNSample;
    const float  gridWidth   = neighbourSize;
    const float  halfWidth   = gridWidth/2.0f;
    const float  stepSize    = gridWidth * __frcp_rn( float(gridNSample-1) );

    if( not first_iteration ) {
        // the best center is located in point_buffer[0]
        center = point_buffer[0].point;
        // @lilian: why do we need this "center = mT * center" in every iteration?
    }
    mT.condition( center );

    const int i = blockIdx.x * 32 + threadIdx.x;
    const int j = blockIdx.y * 32 + threadIdx.y;
    if( i >= gridNSample ) return;
    if( j >= gridNSample ) return;

    const int idx = j * gridNSample + i;


    NearbyPoint* nPoint = &point_buffer[idx];
    identification::CutSignals* signals = &sig_buffer[idx * vCutsSize];

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


#if 1
    const dim3 block( 32, // we use this to sum up signals
                      32, // 32 cuts in one block
                      1 );
    const dim3 grid(  grid_divide( vCutsSize, 32 ), // ceil(#cuts/32) blocks needed
                      1,
                      1 );
#else
    const dim3 block( 32, 1, 1 ); // we use this to sum up signals
    const dim3 grid( vCutsSize, 1, 1 );
#endif

#if 0
    cerr << "GPU: #vCuts=" << vCutsSize << " vCutMaxLen=" << vCutMaxVecLen
         << " grid=(" << grid.x << "," << grid.y << "," << grid.z << ")"
         << " block=(" << block.x << "," << block.y << "," << block.z << ")"
         << endl;
#endif

    popart::identification::idGetSignals
        <<<grid,block>>>
        ( tagIndex,
          nPoint,
          src,
          cut_buffer,
          signals,
          vCutsSize );
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
        }
    }
}

} // namespace identification

__host__
float Frame::idCostFunction(
    const int                           tagIndex, // in - determines index in cut structure
    int                                 iterations,
    const popart::geometry::ellipse&    ellipse,
    const float2                        center,
    const int                           vCutSize,
    // const std::vector<cctag::ImageCut>& vCuts,
    float                               currentNeighbourSize,
    float2&                             bestPointOut,
    popart::geometry::matrix3x3&        bestHomographyOut,
    const cctag::Parameters&            params,
    NearbyPoint*                        cctag_pointer_buffer )
{
    const size_t gridNSample   = params._imagedCenterNGridSample;

    /* reusing various image-sized plane */
    NearbyPoint*                 point_buffer;
    identification::CutStruct*   cut_buffer;
    identification::CutSignals*  signal_buffer;

    point_buffer  = getNearbyPointBuffer();
    cut_buffer    = getCutStructBuffer();
    signal_buffer = getSignalBuffer();

    // should this ellipse change?
    popart::geometry::matrix3x3 mT;
    ellipse.makeConditionerFromEllipse( mT );
    popart::geometry::matrix3x3 mInvT;
    mT.invert( mInvT ); // note: returns false if it fails

    popart::geometry::ellipse transformedEllipse;
    ellipse.projectiveTransform( mInvT, transformedEllipse );

    bool first_iteration = true;

    // cerr << "TagIndex: " << tagIndex << " number of loops: " << iterations << endl;

    for( ; iterations>0; iterations-- ) {
        float neighSize = currentNeighbourSize * max( transformedEllipse.a(),
                                                      transformedEllipse.b() );

        dim3 block( 32, 32, 1 );
        dim3 grid( grid_divide( gridNSample, 32 ),
                   grid_divide( gridNSample, 32 ),
                   1 );

        popart::identification::getSignalsAllNearbyPoints
            <<<grid,block,0,_stream>>>
            ( tagIndex,
              first_iteration,
              _d_plane,
              ellipse,
              mT,
              // mInvT,
              center,
              vCutSize,
              neighSize,
              point_buffer,
              cut_buffer,
              signal_buffer );
        POP_CHK_CALL_IFSYNC;

        dim3 id_block( 32, // we use this to sum up signals
                       32, // we can use some shared memory/warp magic for summing
                       1 );
#ifdef NO_ATOMIC
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
            <<<id_grid,id_block,0,_stream>>>
            ( tagIndex, point_buffer, cut_buffer, signal_buffer, vCutSize );

        /* We search for the minimum of gridNSample x gridNSample
         * nearby points. Default for gridNSample is 5.
         * It is therefore most efficient to use a single-warp kernel
         * for the search.
         */
        const int gridSquare = gridNSample * gridNSample;

        if( gridSquare < 32 ) {
            POP_CHK_CALL_IFSYNC;
            popart::identification::idBestNearbyPoint31max
                <<<1,32,0,_stream>>>
                  ( point_buffer, gridSquare );
            POP_CHK_CALL_IFSYNC;
        } else {
            POP_CHK_CALL_IFSYNC;
            popart::identification::idBestNearbyPoint32plus
                <<<1,32,0,_stream>>>
                  ( point_buffer, gridSquare );
            POP_CHK_CALL_IFSYNC;
        }

        currentNeighbourSize /= (float)((gridNSample-1)/2) ;

        first_iteration = false;
    }

    /* When this kernel finishes, the best point does not
     * exist or it is stored in point_buffer[0]
     */
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( cctag_pointer_buffer,
                                   point_buffer,
                                   sizeof(popart::NearbyPoint),
                                   _stream );
    POP_CHK_CALL_IFSYNC;

    cudaStreamSynchronize( _stream );
    if( not cctag_pointer_buffer->readable ) {
        return FLT_MAX;
    }
    bestPointOut      = cctag_pointer_buffer->point;
    bestHomographyOut = cctag_pointer_buffer->mHomography;
    return cctag_pointer_buffer->result / cctag_pointer_buffer->resSize;
}

__host__
bool Frame::imageCenterOptLoop(
    const int                           tagIndex,     // in - determines index in cut structure
    const popart::geometry::ellipse&    outerEllipse, // in
    float2&                             center,       // in-out
    const int                           vCutSize,
    popart::geometry::matrix3x3&        bestHomographyOut,
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

    float  residual;
    float2 bestPointOut;
    residual = idCostFunction( tagIndex,
                               iterations,
                               outerEllipse,
                               center,
                               vCutSize,
                               neighbourSize,
                               bestPointOut,
                               bestHomographyOut,
                               params,
                               cctag_pointer_buffer );

    if( residual == FLT_MAX ) return false;
    center        =  bestPointOut;
    neighbourSize /= (float)((gridNSample-1)/2) ;
    return true;
}

__host__
size_t Frame::getCutStructBufferByteSize( ) const
{
    /* these are uint8_t */
    return _d_mag.rows * _d_mag.step;
}

__host__
identification::CutStruct* Frame::getCutStructBuffer( ) const
{
    return reinterpret_cast<identification::CutStruct*>( _d_mag.data );
}

__host__
identification::CutStruct* Frame::getCutStructBufferHost( ) const
{
    return reinterpret_cast<identification::CutStruct*>( _h_mag.data );
}

__host__
size_t Frame::getNearbyPointBufferByteSize( ) const
{
    /* these are uint32_t */
    return _d_map.rows * _d_map.step;
}

__host__
NearbyPoint* Frame::getNearbyPointBuffer( ) const
{
    return reinterpret_cast<NearbyPoint*>( _d_map.data );
}

__host__
size_t Frame::getSignalBufferByteSize( ) const
{
    /* these are float */
    return _d_intermediate.rows * _d_intermediate.step;
}

__host__
identification::CutSignals* Frame::getSignalBuffer( ) const
{
    return reinterpret_cast<identification::CutSignals*>( _d_intermediate.data );
}

__host__
void Frame::clearSignalBuffer( )
{
#ifdef DEBUG_FRAME_UPLOAD_CUTS
    if( _d_intermediate.step != _h_intermediate.step ||
        _d_intermediate.rows != _h_intermediate.rows ) {
        cerr << "intermediate dimensions should be identical on host and dev"
             << endl;
        exit( -1 );
    }
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_MEMSET_ASYNC( _d_intermediate.data,
                           -1,
                           _h_intermediate.step * _h_intermediate.rows,
                           _stream );
    POP_CHK_CALL_IFSYNC;
#endif // DEBUG_FRAME_UPLOAD_CUTS
}

}; // namespace popart

