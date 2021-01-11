// 系统头文件
#include <stdlib.h>
#include <stdio.h>

// cuda头文件
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define N 5

__global__ void matrix_multiplication(float* mat_a, float* mat_b, float* mat_c, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float pvalue = 0, m, n;
    for (int k = 0; k < width; ++k) {
        m = mat_a[ty * width + k];
        n = mat_b[k * width + tx];
        pvalue += m * n;
    }
    mat_c[ty * width + tx] = pvalue;
}


// 主机端主函数
int main(void) {
    float* mat_a, * mat_b, * mat_c, * device_mat_a, * device_mat_b, * device_mat_c;
    int mem_size = N * N * sizeof(float);

    // 在主机内存申请 A，B，C 向量的空间
    mat_a = (float*)malloc(mem_size);
    mat_b = (float*)malloc(mem_size);
    mat_c = (float*)malloc(mem_size);

    // 在 GPU 设备申请 A，B，C 向量的空间
    cudaMalloc((void**)&device_mat_a, mem_size);
    cudaMalloc((void**)&device_mat_b, mem_size);
    cudaMalloc((void**)&device_mat_c, mem_size);

    // 初始化主机内存的 A，B 向量
    for (int i = 0; i < N * N; i++) {
        mat_a[i] = 1;
        mat_b[i] = 2;
    }

    // 拷贝主机内存的 A，B 的内容到 GPU 设备的 A，B
    cudaMemcpy(device_mat_a, mat_a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat_b, mat_b, mem_size, cudaMemcpyHostToDevice);

    // GPU 内核函数的维度参数
    dim3 threadsPreBlock(N, N);
    dim3 blocksPreGrid(1);

    // 执行 GPU 内核函数
    matrix_multiplication << < blocksPreGrid, threadsPreBlock >> > (device_mat_a, device_mat_b, device_mat_c, N);

    // 从 GPU 设备复制结果向量 C 到主机内存的 C
    cudaMemcpy(mat_c, device_mat_c, mem_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f\t", mat_c[i * N + j]);
        }
        printf("\n");
    }

    free(mat_a);
    free(mat_b);
    free(mat_c);
    cudaFree(device_mat_a);
    cudaFree(device_mat_b);
    cudaFree(device_mat_c);
}
