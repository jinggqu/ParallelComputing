
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//#define N 1024
__global__ void calcSum(int *d_arr) {
    extern __shared__ float partialSum[];
    unsigned int t = threadIdx.x;
    unsigned int bid = blockIdx.x;
    maxArray[t] = d_arr[t + bid * 512];
    for (unsigned int stride = blockDim.x / 2; stride > 0;  stride >>= 1) {
        __syncthreads();
        if (t < stride)
        maxArray[t] = maxArray[t + stride] > maxArray[t] ? 
                      maxArray[t + stride] : maxArray[t];
    }
}

int main() {
    cudaError_t cudaStatus = cudaSuccess;
    int *h_arr,*d_arr; 
    int N;
    printf("Please input array size: ");
    scanf("%d", &N);
    h_arr= (int *)malloc(sizeof(int)* N * N);
    
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_arr[i * N + j] = i * N + j;
		}
	}
        
    cudaStatus = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // Launch a kernel on the GPU with one thread for each element.
    calcSum <<<N, N, N * N * sizeof(int)>>> (d_arr);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcSum failed!");
		return 1;
	}
	
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
	}
	
    for(int i = 0; i < N; i++){
		printf("The max number of row %d :%d\n", i,  h_arr[i]);
	}
	
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}

