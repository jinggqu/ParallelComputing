#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 1024

__global__ void get_max_value_of_row(float* array, float* max_array) {
    int t = threadIdx.x;
    int bid = blockIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0)
            array[t + (bid % N) * N] = array[t + (bid % N) * N + stride] > array[t + (bid % N) * N] ?
                                       array[t + (bid % N) * N + stride] : array[t + (bid % N) * N];
    }
    max_array[bid % N] = array[t + (bid % N) * N];
}

int main() {
    float* array, * device_array, * max_array, * device_max_array;
    int total = N * N;
    int mem_size = total * sizeof(float);

    array = (float*)malloc(mem_size);
    max_array = (float*)malloc(N * sizeof(float));
    memset(max_array, 0, N);

    for (int i = 0; i < N * N; i++) {
        array[i] = i + 1;
    }

    cudaMalloc((void**)&device_array, mem_size);
    cudaMalloc((void**)&device_max_array, N * sizeof(float));
    cudaMemcpy(device_array, array, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_max_array, max_array, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(N);

    get_max_value_of_row << <blocksPerGrid, threadsPerBlock >> > (device_array, device_max_array);
    cudaMemcpy(max_array, device_max_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("The max number of row %d :%.f\n", i, max_array[i]);
    }

    free(array);
    free(max_array);
    cudaFree(device_array);
    cudaFree(device_max_array);
    return 0;
}

