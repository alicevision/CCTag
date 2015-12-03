#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <cub/cub.cuh>

#define NUMELEMS 1000000

using namespace std;

__constant__
int comparator[10] = { 110, 120, 130, 140, 150, 160, 170, 180, 190, 200 };

struct Bigger_than_500
{
    __device__
    bool operator()(const int &a) const {
        // return ( a < comparator[a%10] );
        return ( a > comparator[a%10] );
    }
};

int main( )
{
    void* ptr;
    int*  ints;
    int*  cuda_ints[3];
    int*  cuda_ct;
    cudaError_t err;

    err = cudaMallocHost( &ptr, sizeof(int)*NUMELEMS);
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    ints = (int*)ptr;

    for( int i=0; i<3; i++ ) {
        err = cudaMalloc( &ptr, sizeof(int)*NUMELEMS);
        if( err != cudaSuccess ) {
            cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
            exit( -1 );
        }
        cuda_ints[i] = (int*)ptr;
    }

    err = cudaMalloc( &ptr, sizeof(int) );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    cuda_ct = (int*)ptr;

    for( int i=0; i<NUMELEMS; i++ ) {
        ints[i] = random() % 500;
    }

    err = cudaMemcpy( cuda_ints[0], ints, sizeof(int)*NUMELEMS, cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }

    size_t interm_sz = 0;
    cub::DoubleBuffer<int> keys( cuda_ints[0], cuda_ints[1] );
    err = cub::DeviceRadixSort::SortKeys<int>( 0,
                                               interm_sz,
                                               keys,
                                               NUMELEMS,
                                               0,             // begin_bit
                                               sizeof(int)*8, // end_bit
                                               0,
                                               true );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    if( interm_sz > NUMELEMS*sizeof(int) ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    err = cub::DeviceRadixSort::SortKeys<int>( cuda_ints[2],
                                               interm_sz,
                                               keys,
                                               NUMELEMS,
                                               0,             // begin_bit
                                               sizeof(int)*8, // end_bit
                                               0,
                                               true );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }

    if( cuda_ints[0] != keys.Current() ) {
        std::swap( cuda_ints[0], cuda_ints[1] );
    }

#if 0
    const int* iptr;
    err = cudaGetSymbolAddress( &ptr, comparator );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    iptr = (const int*)ptr;
#endif

    Bigger_than_500 bigger_than_500;

    interm_sz = 0;
    err = cub::DeviceSelect::If(
        0,
        interm_sz,
        cuda_ints[0],
        cuda_ints[1],
        cuda_ct,
        NUMELEMS,
        bigger_than_500,
        0, true );

    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    if( interm_sz > NUMELEMS*sizeof(int) ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    err = cub::DeviceSelect::If(
        cuda_ints[2],
        interm_sz,
        cuda_ints[0],
        cuda_ints[1],
        cuda_ct,
        NUMELEMS,
        bigger_than_500,
        0, true );

    err = cudaMemcpy( ints, keys.Current(), sizeof(int)*NUMELEMS, cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    cout << "At the end: " << endl;
    for( int i=0; i<1000; i++ ) {
        cout << ints[i] << " ";
    }
    err = cudaMemcpy( ints, cuda_ints[1], sizeof(int)*NUMELEMS, cudaMemcpyDeviceToHost );
    if( err != cudaSuccess ) {
        cerr << __FILE__<< ":" << __LINE__ << " Error: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    }
    cout << endl;
    cout << "Unique: " << endl;
    for( int i=0; i<1000; i++ ) {
        cout << ints[i] << " ";
    }
    cout << endl;
}

