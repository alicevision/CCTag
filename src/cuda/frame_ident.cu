#include <cuda_runtime.h>

#include "frame.h"
#include "clamp.h"
#include "geom_matrix.h"
#include "geom_projtrans.h"

using namespace std;

namespace popart {

namespace identification {

struct CutStruct
{
    float2 start;
    float2 stop;
    float  beginSig;
    float  endSig;
    int    sigSize;
};

struct NearbyPoint
{
    float2 point;
    float  result;
    int    resSize;
    bool   readable;

    /* These homographies are computed once for each NearbyPoint,
     * and used for all of its Cuts. The best one must be returned.
     */
    popart::geometry::matrix3x3 mHomography;
    popart::geometry::matrix3x3 mInvHomography;
};

struct CutSignals
{
    uint32_t outOfBounds;
    float    sig[127];
};

__device__
inline float getPixelBilinear( cv::cuda::PtrStepSzb src, float2 xy )
{
    int px = (int)xy.x; // floor of x
    int py = (int)xy.y; // floor of y
#if 0
    if( px != clamp( px, src.cols-1 ) ) {
        printf("Should clamp px from %d to %d\n", px, clamp( px, src.cols-1 ) );
    }
    if( py != clamp( py, src.rows-1 ) ) {
        printf("Should clamp py from %d to %d\n", py, clamp( py, src.rows-1 ) );
    }
    px = clamp( px, src.cols-1 );
    py = clamp( py, src.rows-1 );
#endif

    // uint8_t p0 = src.ptr(py  )[px  ];
    uint8_t p1 = src.ptr(py  )[px  ];
    uint8_t p2 = src.ptr(py  )[px+1];
    uint8_t p3 = src.ptr(py+1)[px  ];
    uint8_t p4 = src.ptr(py+1)[px+1];

    // Calculate the weights for each pixel
    float fx  = xy.x - (float)px;
    float fy  = xy.y - (float)py;
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;

    float w1 = fx1 * fy1;
    float w2 = fx  * fy1;
    float w3 = fx1 * fy;
    float w4 = fx  * fy;

    // Calculate the weighted sum of pixels (for each color channel)
    return ( p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4 ) / 2.0f;
}

__device__
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

#if 0
// now included in the nearby point dispatcher
__global__
void idMakeHomographies( popart::geometry::ellipse ellipse,
                         const float2 center,
                         popart::geometry::matrix3x3* dev_homographies )
{
    // m1: popart::geometry::matrix3x3 mHomography;
    // m2: popart::geometry::matrix3x3 mInvHomography;

    popart::geometry::matrix3x3& m1 = dev_homographies[0];
    popart::geometry::matrix3x3& m2 = dev_homographies[1];
    ellipse.computeHomographyFromImagedCenter( center, m1 );
    m1.invert( m2 );
}
#endif

/* We use a pair of homographies to extract signals (<=128) for every
 * cut, and every nearby point of a potential center.
 * vCutMaxVecLen is a safety variable, could actually be 100.
 * The function is called from
 *     idNearbyPointDispatcher
 * once for every 
 */
__global__
void idGetSignals( NearbyPoint*         nPoint,
                   cv::cuda::PtrStepSzb src,
                   const CutStruct*     cuts,
                   CutSignals*          signals,
                   const int            vCutsSize,
                   const int            vCutMaxVecLen )
{
    int myCutIdx = blockIdx.x * 32 + threadIdx.y;

    if( myCutIdx >= vCutsSize ) {
        return; // out of bounds
    }

    const CutStruct& myCut     = cuts[myCutIdx];
    CutSignals*      mySignals = &signals[myCutIdx];

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
void idComputeResult( NearbyPoint*      nPoint,
                      const CutStruct*  vCuts,
                      const CutSignals* allcut_signals,
                      const int         vCutsSize,
                      const int         vCutMaxVecLen )
{
    int myPair = blockIdx.x * 32 + threadIdx.y;
    int j      = __float2int_rd( 1.0f + __fsqrt_rd(1.0f+8.0f*myPair) ) / 2;
    int i      = myPair - j*(j-1)/2;

    int   ct   = 0;
    float val  = 0.0f;
    bool  comp = true;

    comp = ( j < vCutsSize && i < j );

    if( comp ) {
        const CutSignals* l_signals = &allcut_signals[i];
        const CutSignals* r_signals = &allcut_signals[j];
        comp  = ( threadIdx.x < vCutMaxVecLen ) &&
                  not l_signals->outOfBounds &&
                  not r_signals->outOfBounds;
        if( comp ) {
            const int limit = vCuts[i].sigSize; // we could also use j ?
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
            atomicAdd( &nPoint->result,  val );
            atomicAdd( &nPoint->resSize, ct );
        }
    }
}

__global__
void idNearbyPointDispatcher( FrameMetaPtr                       meta,
                              cv::cuda::PtrStepSzb               src,
                              const popart::geometry::ellipse  & ellipse,
                              const popart::geometry::matrix3x3& mInvT,
                              float2                             center,
                              const int                          vCutsSize,
                              const int                          vCutMaxVecLen,
                              const float                        neighbourSize,
                              const size_t                       gridNSample,
                              identification::NearbyPoint*       point_buffer,
                              const identification::CutStruct*   cut_buffer,
                              identification::CutSignals*        sig_buffer )
{
    const float gridWidth = neighbourSize;
    const float halfWidth = gridWidth/2.0f;
    const float stepSize  = gridWidth * __frcp_rn( float(gridNSample-1) );

    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    if( i >= gridNSample ) return;
    if( j >= gridNSample ) return;

#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    atomicAdd( &meta.num_nearby_points(), 1);
#endif

    int idx = j * gridNSample + i;

    float2 point = make_float2( center.x - halfWidth + i*stepSize,
                                center.y - halfWidth + j*stepSize );
    mInvT.condition( point );

    identification::NearbyPoint* nPoint  = &point_buffer[idx];

    nPoint->point = point;

    // GRIFF: is this multiplication correct???
    identification::CutSignals*  signals = &sig_buffer[idx * vCutsSize];

    nPoint->result   = 0.0f;
    nPoint->resSize  = 0;
    nPoint->readable = true;
    ellipse.computeHomographyFromImagedCenter( nPoint->point, nPoint->mHomography );
    nPoint->mHomography.invert( nPoint->mInvHomography );

    dim3 block( 32, // we use this to sum up signals
                32, // we can use some shared memory/warp magic for summing
                1 );
    dim3 grid(  grid_divide( vCutsSize, 32 ), 1, 1 );

#if 0
    cerr << "GPU: #vCuts=" << vCutsSize << " vCutMaxLen=" << vCutMaxVecLen
         << " grid=(" << grid.x << "," << grid.y << "," << grid.z << ")"
         << " block=(" << block.x << "," << block.y << "," << block.z << ")"
         << endl;
#endif

    popart::identification::idGetSignals
        <<<grid,block>>>
        ( nPoint,
          src,
          cut_buffer,
          signals,
          vCutsSize,
          vCutMaxVecLen );

    int numPairs = vCutsSize*(vCutsSize-1)/2;
    block.x = 32; // we use this to sum up signals
    block.y = 32; // we can use some shared memory/warp magic for summing
    block.z = 1;
    grid.x  = grid_divide( numPairs, 32 );
    grid.y  = 1;
    grid.z  = 1;

    // _meta.toDevice( Identification_result, 0.0f, _stream );
    // _meta.toDevice( Identification_resct,  0,    _stream );

    popart::identification::idComputeResult
        <<<grid,block>>>
        ( nPoint, cut_buffer, signals, vCutsSize, vCutMaxVecLen );
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
void Frame::uploadCuts( const std::vector<cctag::ImageCut>& vCuts )
{
    identification::CutStruct* csptr = getCutStructBufferHost();

    std::vector<cctag::ImageCut>::const_iterator vit  = vCuts.begin();
    std::vector<cctag::ImageCut>::const_iterator vend = vCuts.end();
    for( ; vit!=vend; vit++ ) {
        csptr->start.x     = vit->start().getX();
        csptr->start.y     = vit->start().getY();
        csptr->stop.x      = vit->stop().getX();
        csptr->stop.y      = vit->stop().getY();
        csptr->beginSig    = vit->beginSig();
        csptr->endSig      = vit->endSig();
        csptr->sigSize     = vit->imgSignal().size();
        csptr++;
    }

    POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( getCutStructBuffer(),
                                     getCutStructBufferHost(),
                                     vCuts.size() * sizeof(identification::CutStruct),
                                     _stream );
}

__host__
double Frame::idCostFunction( const popart::geometry::ellipse&    ellipse,
                              const float2                        center,
                              const std::vector<cctag::ImageCut>& vCuts,
                              const size_t                        vCutMaxVecLen,
                              float                               neighbourSize,
                              const size_t                        gridNSample,
                              float2&                             bestPointOut,
                              popart::geometry::matrix3x3&        bestHomographyOut )
{
#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    _meta.toDevice( Num_nearby_points, 0, _stream );
#endif

    const size_t g = gridNSample * gridNSample;
    if( g*sizeof(identification::NearbyPoint) > getNearbyPointBufferByteSize() ) {
        cerr << __FILE__ << ":" << __LINE__
             << "ERROR: re-interpreted image plane too small to hold point search rsults" << endl;
        exit( -1 );
    }

    if( vCuts.size() * sizeof(identification::CutStruct) > getCutStructBufferByteSize() ) {
        cerr << __FILE__ << ":" << __LINE__
             << "ERROR: re-interpreted image plane too small to hold all intermediate homographies" << endl;
        exit( -1 );
    }

    clearSignalBuffer( );
    uploadCuts( vCuts );

    /* reusing various image-sized plane */
    identification::NearbyPoint* point_buffer;
    identification::CutStruct*   cut_buffer;
    identification::CutSignals*  signal_buffer;

    point_buffer  = getNearbyPointBuffer();
    cut_buffer    = getCutStructBuffer();
    signal_buffer = getSignalBuffer();

    popart::geometry::matrix3x3 mT;
    ellipse.makeConditionerFromEllipse( mT );
    popart::geometry::matrix3x3 mInvT;
    mT.invert( mInvT ); // note: returns false if it fails

    popart::geometry::ellipse transformedEllipse;
    ellipse.projectiveTransform( mInvT, transformedEllipse );

    neighbourSize *= max( transformedEllipse.a(),
                          transformedEllipse.b() );

    float2 condCenter = center;
    mT.condition( condCenter );

    dim3 block( 32, 32, 1 );
    dim3 grid( grid_divide( gridNSample, 32 ),
               grid_divide( gridNSample, 32 ),
               1 );

    popart::identification::idNearbyPointDispatcher
        <<<grid,block,0,_stream>>>
        ( _meta,
          _d_plane,
          ellipse,
          mInvT,
          condCenter,
          vCuts.size(),
          vCutMaxVecLen,
          neighbourSize,
          gridNSample,
          point_buffer,
          cut_buffer,
          signal_buffer );

#ifdef CPU_GPU_COST_FUNCTION_COMPARE
    int aNumber;
    _meta.fromDevice( Num_nearby_points, aNumber, _stream );
    cudaStreamSynchronize( _stream );
    std::cerr << "Number of nearby points on device: " << aNumber << std::endl;
#endif

    /* We search for the minimum of gridNSample x gridNSample
     * nearby points. Default for gridNSample is 5.
     * It is therefore most efficient to use a single-warp kernel
     * for the search.
     */
    block.x = 32;
    block.y = 1;
    block.z = 1;
    grid.x  = 1;
    grid.y  = 1;
    grid.z  = 1;
    int gridSquare = gridNSample * gridNSample;

    if( gridSquare < 32 ) {
        popart::identification::idBestNearbyPoint31max
            <<<grid,block,0,_stream>>>
            ( point_buffer, gridSquare );
    } else {
        popart::identification::idBestNearbyPoint32plus
            <<<grid,block,0,_stream>>>
            ( point_buffer, gridSquare );
    }

    /* When this kernel finishes, the best point does not
     * exist or it is stored in point_buffer[0]
     */
    popart::identification::NearbyPoint point;
    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &point,
                                   point_buffer,
                                   sizeof(popart::identification::NearbyPoint),
                                   _stream );
#warning this copy function is blocking

    cudaStreamSynchronize( _stream );
    if( point.readable ) {
        bestPointOut      = point.point;
        bestHomographyOut = point.mHomography;
        return point.result / point.resSize;
    } else {
        return FLT_MAX;
    }
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
identification::NearbyPoint* Frame::getNearbyPointBuffer( ) const
{
    return reinterpret_cast<identification::NearbyPoint*>( _d_map.data );
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
    POP_CUDA_MEMSET_ASYNC( _d_intermediate.data,
                           -1,
                           _h_intermediate.step * _h_intermediate.rows,
                           _stream );
#endif // DEBUG_FRAME_UPLOAD_CUTS
}

}; // namespace popart

