
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 1024

__global__ void getMaxValueOfRow(float *d_arr, float *maxArray) {
    int t = threadIdx.x;
    int bid = blockIdx.x;
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0)
            d_arr[t + (bid % N) * N] = d_arr[t + (bid % N) * N + stride] > d_arr[t + (bid % N) * N] ? 
                                       d_arr[t + (bid % N) * N + stride] : d_arr[t + (bid % N) * N];
    }
    
    maxArray[bid % N] = d_arr[t + (bid % N) * N];
}

int main() {
    float *h_arr, *d_arr, *h_maxArray, *d_maxArray; 
    int total = N * N;
    int mem_size = total * sizeof(float);

    h_arr = (float *) malloc(mem_size);
    h_maxArray = (float *) malloc(N * sizeof(float));
    memset(h_maxArray, 0, N);

    for (int i = 0; i < N * N; i++) {
        h_arr[i] = i + 1;
    }

    cudaMalloc((void **) &d_arr, mem_size);
    cudaMalloc((void **) &d_maxArray, N * sizeof(float));
    cudaMemcpy(d_arr, h_arr, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxArray, h_maxArray, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(N);
    getMaxValueOfRow <<<blocksPerGrid, threadsPerBlock>>> (d_arr, d_maxArray);

    cudaMemcpy(h_maxArray, d_maxArray, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < N; i++){
        printf("The max number of row %d :%.f\n", i,  h_maxArray[i]);
    }
    
    free(h_arr);
    free(h_maxArray);
    cudaFree(d_arr);
    cudaFree(d_maxArray);
    
    return 0;
}

