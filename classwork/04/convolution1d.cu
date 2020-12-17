#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void convolution_1D_basic_kernel(float* in, float* mask, float* out, int mask_size, int input_size) {

    float val = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_start_point = i - (mask_size / 2);

    if (i < input_size) {
        for (int j = 0; j < mask_size; j++) {
            if ((n_start_point + j) >= 0 && (n_start_point + j < input_size)) {
                val += in[n_start_point + j] * mask[j];
            }
        }
        out[i] = val;
    }
}


int main() {
    int input_size = 7, mask_size = 5;
    float* host_in, * host_out, * device_in, * device_mask, * device_out;
    float host_mask[5] = { 3, 4, 5, 4, 3 };

    host_in = (float*)malloc(input_size * sizeof(float));
    host_out = (float*)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; i++) {
        host_in[i] = i + 1;
    }

    cudaMalloc((void**)&device_in, input_size * sizeof(float));
    cudaMalloc((void**)&device_mask, mask_size * sizeof(float));
    cudaMalloc((void**)&device_out, input_size * sizeof(float));
    cudaMemcpy(device_in, host_in, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mask, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(input_size);
    dim3 blocksPerGrid(1);

    convolution_1D_basic_kernel <<< blocksPerGrid, threadsPerBlock >>> (device_in, device_mask, device_out, mask_size, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(host_out, device_out, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_size; ++i) {
        printf("%.f\t", host_out[i]);
    }
    printf("\n");

    cudaFree(device_in);
    cudaFree(device_mask);
    cudaFree(device_out);
    free(host_in);
    free(host_out);
    return 0;
}

