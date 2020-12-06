#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 512

__global__ void getMaxValueOfRow(float *d_arr, float *maxArray) {
    unsigned int t = threadIdx.x;
    unsigned int bid = blockIdx.x;
    
    maxArray[t] = d_arr[t + bid * N];
    for (unsigned int stride = blockDim.x / 2; stride > 0;  stride >>= 1) {
        __syncthreads();
        if (t < stride)
            maxArray[t] = maxArray[t + stride] > maxArray[t] ? maxArray[t + stride] : maxArray[t];
    }
}

int main() {
    float *h_arr, *d_arr, *h_maxArray, *d_maxArray;
    int total = N * N;
    int mem_size = total * sizeof(float);

    h_arr = (float *) malloc(mem_size);
    h_maxArray = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < total; i++) {
        h_arr[i] = 3.0;
    }

    for (int i = 0; i < N; i++) {
        h_maxArray[i] = 0.0;
    }

    cudaMalloc((void **) &d_arr, mem_size);
    cudaMalloc((void **) &d_maxArray, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxArray, h_maxArray, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(N);

    // 记录程序开始运行的时间
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    getMaxValueOfRow <<< blocksPerGrid, threadsPerBlock >>> (d_arr, d_maxArray);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time elapsed: %.6f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_maxArray, d_maxArray, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; ++i) {
    //     printf("The max number of row %d :%.f\n", i,  h_maxArray[i]);
    // }

    // 验证结果
    int count = 0;
    for (int i = 0; i < N * N; ++i) {
        if (h_maxArray[i] == 3)
            count++;
    }
    printf("count = %d\n", count);
    
    cudaFree(d_arr);
    cudaFree(d_maxArray);
    free(h_arr);
    free(h_maxArray);
    return 0;
}

