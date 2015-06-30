#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

#undef ABBREVIATED_HYSTERESIS

namespace popart
{

using namespace std;

namespace hysteresis
{
__shared__ uint8_t array[34][34];

__device__
uint8_t get( const cv::cuda::PtrStepSzb map, const int idx, const int idy )
{
    return map.ptr( clamp( idy, map.rows ) )[ clamp( idx, map.cols ) ];
}

__device__
void set( cv::cuda::PtrStepSzb map, const int idx, const int idy, uint8_t val )
{
    if( outOfBounds( idx, idy, map ) ) return;
    map.ptr(idy)[idx] = val;
}

__device__
void loadHoriz( const cv::cuda::PtrStepSzb map, int offy, const int idx, const int idy )
{
    int offx = threadIdx.x + 1;
    array[offy][offx] = get( map, idx, idy );

    if( threadIdx.x == 0 )
        array[offy][offx-1] = get( map, idx-1, idy );
    if( threadIdx.x == 31 )
        array[offy][offx+1] = get( map, idx+1, idy );
}

__device__
void loadVert( const cv::cuda::PtrStepSzb map, int offx, const int idx, const int idy )
{
    int offy = threadIdx.x + 1;
    array[offy][offx] = get( map, idx, idy );
}

__device__
void load( const cv::cuda::PtrStepSzb map )
{
    const int idx  = blockIdx.x * 32 + threadIdx.x;
    const int idy  = blockIdx.y * 32 + threadIdx.y;
    const int offx = threadIdx.x + 1;
    const int offy = threadIdx.y + 1;

    array[offy][offx] = get( map, idx, idy );

    if( threadIdx.y == 0 ) {
        loadHoriz( map,  0, idx, blockIdx.y*32- 1 );
        loadHoriz( map, 33, idx, blockIdx.y*32+32 );
        loadVert(  map,  0, blockIdx.x*32- 1, blockIdx.y*32+threadIdx.x );
        loadVert(  map, 33, blockIdx.x*32+32, blockIdx.y*32+threadIdx.x );
    }
    __syncthreads();
}

__device__
bool update( )
{
    const int idx = threadIdx.x + 1;
    const int idy = threadIdx.y + 1;

    uint8_t val = array[idy][idx];

    if( val == 1 ) {
        int n = ( array[idy-1][idx-1] == 2 )
              + ( array[idy  ][idx-1] == 2 )
              + ( array[idy+1][idx-1] == 2 )
              + ( array[idy-1][idx  ] == 2 )
              + ( array[idy+1][idx  ] == 2 )
              + ( array[idy-1][idx+1] == 2 )
              + ( array[idy  ][idx+1] == 2 )
              + ( array[idy+1][idx+1] == 2 );
        if( n > 0 ) {
            array[idy][idx] = 2;
            val = 2;
            return true;
        }
    }
    return false;
}
} // namespace hysteresis

__global__
void edge_hysteresis( const cv::cuda::PtrStepSzb map, cv::cuda::PtrStepSzb edges, bool final )
{
    hysteresis::load( map );

    uint8_t val = hysteresis::array[threadIdx.y+1][threadIdx.x+1];

    if( reduceAND_32x32( val == 1 ) ) {
        for( int i=0; i<20; i++ ) {
            bool updated = hysteresis::update( );
            bool any_more = reduceAND_32x32( updated );
            if( not any_more ) break;
        }
    }

    val = hysteresis::array[threadIdx.y+1][threadIdx.x+1];
    val = ( final && val == 1 ) ? 0 : val;
    hysteresis::set( edges, blockIdx.x*32+threadIdx.x, blockIdx.y*32+threadIdx.y, val );
}

__host__
void Frame::applyHyst( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 big_block;
    dim3 big_grid;
    big_block.x = 32;
    big_block.y = 32;
    big_grid.x  = ( getWidth()  / 32 ) + ( getWidth()  % 32 == 0 ? 0 : 1 );
    big_grid.y  = ( getHeight() / 32 ) + ( getHeight() % 32 == 0 ? 0 : 1 );

    edge_hysteresis
        <<<big_grid,big_block,0,_stream>>>
        ( _d_map, cv::cuda::PtrStepSzb(_d_intermediate), false );
    POP_CHK_CALL_IFSYNC;

    edge_hysteresis
        <<<big_grid,big_block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), _d_hyst_edges, true );
    POP_CHK_CALL_IFSYNC;

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart

