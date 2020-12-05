#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define N 1024

__global__ void matrixMultiplication(float *matrixM, float *matrixN, float *matrixP) {
    int bx = blockIdx.x; 
    int tx = threadIdx.x;
    float sum = 0, m, n;
    for (int k = 0; k < N; ++k) {
        m = matrixM[bx * N + k];
        n = matrixN[k * N + tx];
        sum += m * n;
    }
    matrixP[bx * N + tx] = sum;
}

int main(void) {
    float *h_matrixM, *h_matrixN, *h_matrixP, *d_matrixM, *d_matrixN, *d_matrixP;
    int total = N * N;
    int mem_size = total * sizeof(float);

    h_matrixM = (float *) malloc(mem_size);
    h_matrixN = (float *) malloc(mem_size);
    h_matrixP = (float *) malloc(mem_size);
    cudaMalloc((void **) &d_matrixM, mem_size);
    cudaMalloc((void **) &d_matrixN, mem_size);
    cudaMalloc((void **) &d_matrixP, mem_size);

    for (int i = 0; i < total; ++i) {
        h_matrixM[i] = 3;
        h_matrixN[i] = 2;
    }

    cudaMemcpy(d_matrixM, h_matrixM, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixN, h_matrixN, mem_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid(1024);
    
    // 记录程序开始运行的时间
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrixMultiplication <<< blocksPerGrid, threadsPerBlock >>> (d_matrixM, d_matrixN, d_matrixP);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time elapsed: %.6f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_matrixP, d_matrixP, mem_size, cudaMemcpyDeviceToHost);

    // 输出结果
    //for (int i = 0; i < N * N; ++i) {
    //    printf("h_matrixP[%d] = %.6f\n", i, h_matrixP[i]);
    //}

    free(h_matrixM);
    free(h_matrixN);
    free(h_matrixP);
    cudaFree(d_matrixM);
    cudaFree(d_matrixN);
    cudaFree(d_matrixP);
    return 0;
}
