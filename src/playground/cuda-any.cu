#define WIDTH    1920
#define HEIGHT   1200
#define THREAD_X 32
#define THREAD_Y 32
#define BLOCK_X  60 // 1920 / 32
#define BLOCK_Y  38 // 1200 / 32 + ( 1200 % 32 == 0 ? 0 : 1 )


__shared__ volatile uint8_t flip[32][32];

int main()
{
    int      counter;
    uint8_t* hostplane;
    void*    plane;
    void*    intermediate;
    size_t   pitch;

    cudaMallocHost ( (uint8_t*)&hostplane, WIDTH*HEIGHT );
    cudaMallocPitch( &plane, &pitch, WIDTH, HEIGHT );
    cudaMallocPitch( &intermediate, &pitch, WIDTH, HEIGHT );
    cudaMalloc     ( &counter, sizeof(int) );

    for( int row=0; row<HEIGHT; row++ )
        for( int col=0; col<WIDTH; col++ )
            hostplane[row*WIDTH+col] = random() % 3;

    cudaMemcpy2D( plane, pitch,
                  hostplane, WIDTH,
                  WIDTH, HEIGHT,
                  cudaMemcpyHostToDevice );
    cudaMemset2D( intermediate, pitch,
                  0,
                  WIDTH, HEIGHT );
    cudaMemset( counter,
                0,
                sizeof(int) );

    cudaFree( counter );
    cudaFree( intermediate );
    cudaFree( plane );
    cudaFreeHost( hostplane );
}

