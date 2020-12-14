#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void convolution_1D_basic_kernel(float* N, float* M, float* P, int MASK_WIDTH, int Width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int n_start_point = i - (MASK_WIDTH / 2);

    if (i < Width) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            if ((n_start_point + j) >= 0 && (n_start_point + j < Width)) {
                Pvalue += N[n_start_point + j] * M[j];
            }
        }
        P[i] = Pvalue;
    }
}


int main() {
    int nSize = 7, mSize = 5;
    float* host_N, * host_P, * device_N, * device_M, * device_P;
    float host_M[5] = { 3, 4, 5, 4, 3 };

    host_N = (float*)malloc(nSize * sizeof(float));
    host_P = (float*)malloc(nSize * sizeof(float));
    for (int i = 0; i < nSize; i++) {
        host_N[i] = i * 1.0 + 1;
    }

    cudaMalloc((void**)&device_N, nSize * sizeof(float));
    cudaMalloc((void**)&device_M, mSize * sizeof(float));
    cudaMalloc((void**)&device_P, nSize * sizeof(float));
    cudaMemcpy(device_N, host_N, nSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M, host_M, mSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(nSize);
    dim3 blocksPerGrid(1);

    convolution_1D_basic_kernel <<< blocksPerGrid, threadsPerBlock >>> (device_N, device_M, device_P, mSize, nSize);
    cudaDeviceSynchronize();

    cudaMemcpy(host_P, device_P, nSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nSize; ++i) {
        printf("%.f\t", host_P[i]);
    }
    printf("\n");

    cudaFree(device_N);
    cudaFree(device_M);
    cudaFree(device_P);
    free(host_N);
    free(host_P);
    return 0;
}

