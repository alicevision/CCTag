__global__
void compute_map( const PtrStepSz16s dx,
                  const PtrStepSz16s dy,
                  const PtrStepSz32u mag,
                  PtrStepSzb         map,
                  const float        low_thresh,
                  const float        high_thresh )
{
    const int CANNY_SHIFT = 15;
    const int TG22 = (int32_t)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

    const int block_x = blockIdx.x * V7_WIDTH;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    if( idx >= dx.cols ) return;
    if( idy >= dx.rows ) return;

    int32_t  dxVal  = dx.ptr(idy)[idx];
    int32_t  dyVal  = dy.ptr(idy)[idx];
    uint32_t magVal = mag.ptr(idy)[idx];

    // -1 if only is negative, 1 else
    const int signVal = (dxVal ^ dyVal) < 0 ? -1 : 1;

    dxVal = ::abs(dxVal);
    dyVal = ::abs(dyVal);

    // 0 - the pixel can not belong to an edge
    // 1 - the pixel might belong to an edge
    // 2 - the pixel does belong to an edge
    uint8_t edge_type = 0;

    if( magVal > low_thresh )
    {
        const int32_t tg22x = dxVal * TG22;
        const int32_t tg67x = tg22x + ((dxVal + dxVal) << CANNY_SHIFT);

        dyVal <<= CANNY_SHIFT;

        int2 x = (dyVal < tg22x) ? make_int2( idx - 1, idx + 1 )
                                 : (dyVal > tg67x ) ? make_int2( idx, idx )
                                                    : make_int2( idx - signVal, idx + signVal );
        int2 y = (dyVal < tg22x) ? make_int2( idy, idy )
                                 : make_int2( idy - 1, idy + 1 );

        x.x = clamp( x.x, dx.cols );
        x.y = clamp( x.y, dx.cols );
        y.x = clamp( y.x, dx.rows );
        y.y = clamp( y.y, dx.rows );

        if( magVal > mag.ptr(y.x)[x.x] && mavVal >= mag.ptr(y.y)[x.y] ) {
            edge_type = 1 + (int)(magVal > high_thresh);
        }
    }

    map.ptr(idy)[idx] = edge_type;
}


//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    __device__ int counter = 0;

    __device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
    {
        return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
    }

    __global__ void edgesHysteresisLocalKernel(PtrStepSzi map, short2* st)
    {
        __shared__ volatile int smem[18][18];

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        smem[threadIdx.y + 1][threadIdx.x + 1] = checkIdx(y, x, map.rows, map.cols) ? map(y, x) : 0;
        if (threadIdx.y == 0)
            smem[0][threadIdx.x + 1] = checkIdx(y - 1, x, map.rows, map.cols) ? map(y - 1, x) : 0;
        if (threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][threadIdx.x + 1] = checkIdx(y + 1, x, map.rows, map.cols) ? map(y + 1, x) : 0;
        if (threadIdx.x == 0)
            smem[threadIdx.y + 1][0] = checkIdx(y, x - 1, map.rows, map.cols) ? map(y, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1)
            smem[threadIdx.y + 1][blockDim.x + 1] = checkIdx(y, x + 1, map.rows, map.cols) ? map(y, x + 1) : 0;
        if (threadIdx.x == 0 && threadIdx.y == 0)
            smem[0][0] = checkIdx(y - 1, x - 1, map.rows, map.cols) ? map(y - 1, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
            smem[0][blockDim.x + 1] = checkIdx(y - 1, x + 1, map.rows, map.cols) ? map(y - 1, x + 1) : 0;
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][0] = checkIdx(y + 1, x - 1, map.rows, map.cols) ? map(y + 1, x - 1) : 0;
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
            smem[blockDim.y + 1][blockDim.x + 1] = checkIdx(y + 1, x + 1, map.rows, map.cols) ? map(y + 1, x + 1) : 0;

        __syncthreads();

        if (x >= map.cols || y >= map.rows)
            return;

        int n;

        #pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            n = 0;

            if (smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
            {
                n += smem[threadIdx.y    ][threadIdx.x    ] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y    ][threadIdx.x + 2] == 2;

                n += smem[threadIdx.y + 1][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;

                n += smem[threadIdx.y + 2][threadIdx.x    ] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
                n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
            }

            __syncthreads();

            if (n > 0)
                smem[threadIdx.y + 1][threadIdx.x + 1] = 2;

            __syncthreads();
        }

        const int e = smem[threadIdx.y + 1][threadIdx.x + 1];

        map(y, x) = e;

        n = 0;

        if (e == 2)
        {
            n += smem[threadIdx.y    ][threadIdx.x    ] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y    ][threadIdx.x + 2] == 1;

            n += smem[threadIdx.y + 1][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;

            n += smem[threadIdx.y + 2][threadIdx.x    ] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
            n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
        }

        if (n > 0)
        {
            const int ind =  ::atomicAdd(&counter, 1);
            st[ind] = make_short2(x, y);
        }
    }

    void edgesHysteresisLocal(PtrStepSzi map, short2* st1, cudaStream_t stream)
    {
        void* counter_ptr;
        cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, counter) );

        cudaSafeCall( cudaMemsetAsync(counter_ptr, 0, sizeof(int), stream) );

        const dim3 block(16, 16);
        const dim3 grid(divUp(map.cols, block.x), divUp(map.rows, block.y));

        edgesHysteresisLocalKernel<<<grid, block, 0, stream>>>(map, st1);
        cudaSafeCall( cudaGetLastError() );

        if (stream == NULL)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    __constant__ int c_dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    __constant__ int c_dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    __global__ void edgesHysteresisGlobalKernel(PtrStepSzi map, short2* st1, short2* st2, const int count)
    {
        const int stack_size = 512;

        __shared__ int s_counter;
        __shared__ int s_ind;
        __shared__ short2 s_st[stack_size];

        if (threadIdx.x == 0)
            s_counter = 0;

        __syncthreads();

        int ind = blockIdx.y * gridDim.x + blockIdx.x;

        if (ind >= count)
            return;

        short2 pos = st1[ind];

        if (threadIdx.x < 8)
        {
            pos.x += c_dx[threadIdx.x];
            pos.y += c_dy[threadIdx.x];

            if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
            {
                map(pos.y, pos.x) = 2;

                ind = Emulation::smem::atomicAdd(&s_counter, 1);

                s_st[ind] = pos;
            }
        }

        __syncthreads();

        while (s_counter > 0 && s_counter <= stack_size - blockDim.x)
        {
            const int subTaskIdx = threadIdx.x >> 3;
            const int portion = ::min(s_counter, blockDim.x >> 3);

            if (subTaskIdx < portion)
                pos = s_st[s_counter - 1 - subTaskIdx];

            __syncthreads();

            if (threadIdx.x == 0)
                s_counter -= portion;

            __syncthreads();

            if (subTaskIdx < portion)
            {
                pos.x += c_dx[threadIdx.x & 7];
                pos.y += c_dy[threadIdx.x & 7];

                if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
                {
                    map(pos.y, pos.x) = 2;

                    ind = Emulation::smem::atomicAdd(&s_counter, 1);

                    s_st[ind] = pos;
                }
            }

            __syncthreads();
        }

        if (s_counter > 0)
        {
            if (threadIdx.x == 0)
            {
                s_ind = ::atomicAdd(&counter, s_counter);

                if (s_ind + s_counter > map.cols * map.rows)
                    s_counter = 0;
            }

            __syncthreads();

            ind = s_ind;

            for (int i = threadIdx.x; i < s_counter; i += blockDim.x)
                st2[ind + i] = s_st[i];
        }
    }

    void edgesHysteresisGlobal(PtrStepSzi map, short2* st1, short2* st2, cudaStream_t stream)
    {
        void* counter_ptr;
        cudaSafeCall( cudaGetSymbolAddress(&counter_ptr, canny::counter) );

        int count;
        cudaSafeCall( cudaMemcpyAsync(&count, counter_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        cudaSafeCall( cudaStreamSynchronize(stream) );

        while (count > 0)
        {
            cudaSafeCall( cudaMemsetAsync(counter_ptr, 0, sizeof(int), stream) );

            const dim3 block(128);
            const dim3 grid(::min(count, 65535u), divUp(count, 65535), 1);

            edgesHysteresisGlobalKernel<<<grid, block, 0, stream>>>(map, st1, st2, count);
            cudaSafeCall( cudaGetLastError() );

            if (stream == NULL)
                cudaSafeCall( cudaDeviceSynchronize() );

            cudaSafeCall( cudaMemcpyAsync(&count, counter_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream) );
            cudaSafeCall( cudaStreamSynchronize(stream) );

            count = min(count, map.cols * map.rows);

            //std::swap(st1, st2);
            short2* tmp = st1;
            st1 = st2;
            st2 = tmp;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

namespace canny
{
    struct GetEdges : unary_function<int, uchar>
    {
        __device__ __forceinline__ uchar operator ()(int e) const
        {
            return (uchar)(-(e >> 1));
        }

        __host__ __device__ __forceinline__ GetEdges() {}
        __host__ __device__ __forceinline__ GetEdges(const GetEdges&) {}
    };
}

namespace cv { namespace cuda { namespace device
{
    template <> struct TransformFunctorTraits<canny::GetEdges> : DefaultTransformFunctorTraits<canny::GetEdges>
    {
        enum { smart_shift = 4 };
    };
}}}

namespace canny
{
    void getEdges(PtrStepSzi map, PtrStepSzb dst, cudaStream_t stream)
    {
        transform(map, dst, GetEdges(), WithOutMask(), stream);
    }
}

#endif /* CUDA_DISABLER */
