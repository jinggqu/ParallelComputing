#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <limits.h>

#define N 8192


// 对向量每个坐标取平方
__global__ void get_vector_squared(float* vector, float* vector_squared) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    vector_squared[idx] = vector[idx] * vector[idx];
}


// 对坐标平方后的向量累加求和
__global__ void get_vector_sqaured_sum_kernel(float *vector_squared, double *vector_squared_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (unsigned int stride = gridDim.x * blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (idx < stride)
            vector_squared[idx] = vector_squared[idx + stride] + vector_squared[idx];
    }
    *vector_squared_sum = vector_squared[0];
}

// 生成矩阵
__global__ void create_matrix_kernel(float* vector, float* matrix, float vector_mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = 0; i < N; i++) {
        matrix[i + idx * N] = (vector[idx] * vector[i]) / vector_mod;
    }
}

// 获取矩阵中每一行的最大值与最小值，将其保存到对应的数组中
__global__ void get_max_min_kernel(float* matrix, float* min_array, float* max_array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = 0; i < N; i++) {
        max_array[idx] = matrix[i + idx * N] > max_array[idx] ? matrix[i + idx * N] : max_array[idx];
        min_array[idx] = matrix[i + idx * N] < min_array[idx] ? matrix[i + idx * N] : min_array[idx];
    }
}

// 获取向量坐标的最大值
__global__ void get_max_value_of_row(float* array, float* max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = gridDim.x * blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (idx < stride)
            array[idx] = array[idx + stride] > array[idx] ? array[idx + stride] : array[idx];
    }
    *max = array[0];
}

// 获取向量坐标的最小值
__global__ void get_min_value_of_row(float* array, float* min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = gridDim.x * blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (idx < stride)
            array[idx] = array[idx + stride] < array[idx] ? array[idx + stride] : array[idx];
    }
    *min = array[0];
}


void get_max_min(float *vector, float *matrix, float min, float max) {
    double r_square_sum = 0;
    for (int i = 0; i < N; i++) {
        r_square_sum += vector[i] * vector[i];
    }

    double r_mod = sqrtf(r_square_sum);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (vector[i] * vector[j]) / r_mod;
            if (i == 0 && j == 0) {
                max = matrix[0];
                min = matrix[0];
            } else {
                max = matrix[i * N + j] > max ? matrix[i * N + j] : max;
                min = matrix[i * N + j] < min ? matrix[i * N + j] : min;
            }
        }
    }
    printf("CPU => min = %.6f, max = %.6f\n\n", min, max);
}


void init_Data(char* file_path, float* vector) {
    FILE* file;
    file = fopen(file_path, "r");
    for (int i = 0; !feof(file); i++)
        fscanf(file, "%f", &vector[i]);
    fclose(file);
}

int main() {
    float *host_vector, *host_matrix, *host_min_array, *host_max_array;
    float host_min = 0, host_max = 0;
    double vector_mod = 0, vector_squared_sum = 0;

    host_vector = (float*)malloc(sizeof(float) * N);
    host_matrix = (float*)malloc(sizeof(float) * N * N);
    host_min_array = (float*)malloc(sizeof(float) * N);
    host_max_array = (float*)malloc(sizeof(float) * N);
    

    // 读取数据
    init_Data("./testdata6.txt", host_vector);
    

    //========================= CPU start =========================
    // 记录程序开始运行的时间
    double startTime, endTime;
    startTime = (double)clock();

    // CPU 函数
    get_max_min(host_vector, host_matrix, host_min, host_max);

    // 输出程序运行花费的时间
    endTime = (double)clock();
    printf("Time elapsed on CPU: %.6f ms\n\n", endTime - startTime);
    //========================= CPU end ===========================


    //========================= GPU start =========================
    // 记录程序开始运行的时间
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // GPU 函数
    // 初始化变量
    float* device_vector, *device_vector_squared;
    float* device_matrix;
    double *device_vector_squared_sum;
    float* device_min_array, * device_max_array;

    cudaMalloc((void**)&device_vector, sizeof(float) * N);
    cudaMalloc((void**)&device_vector_squared, sizeof(float) * N);
    cudaMalloc((void**)&device_matrix, sizeof(float) * N * N);
    cudaMalloc((void**)&device_vector_squared_sum, sizeof(double));
    cudaMalloc((void**)&device_min_array, sizeof(float) * N);
    cudaMalloc((void**)&device_max_array, sizeof(float) * N);
    cudaMemcpy(device_vector, host_vector, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vector_squared_sum, &vector_squared_sum, sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid(8);

    get_vector_squared <<< blocksPerGrid, threadsPerBlock >>> (device_vector, device_vector_squared);
    cudaDeviceSynchronize();

    get_vector_sqaured_sum_kernel <<< blocksPerGrid, threadsPerBlock >>> (device_vector_squared, device_vector_squared_sum);
    cudaDeviceSynchronize();
    cudaMemcpy(&vector_squared_sum, device_vector_squared_sum, sizeof(double), cudaMemcpyDeviceToHost);


    // 求向量的模
    vector_mod = sqrtf(vector_squared_sum);
    //printf("Modulus of vector = %.6f\n", vector_mod);


    // 生成矩阵
    create_matrix_kernel <<< blocksPerGrid, threadsPerBlock >>> (device_vector, device_matrix, vector_mod);
    cudaDeviceSynchronize();
    cudaMemcpy(host_matrix, device_matrix, sizeof(float) * N * N, cudaMemcpyDeviceToHost);


    // 获取矩阵中每行的最大值与最小值，存储在一个一维数组中
    for (int i = 0; i < N; i++) {
        host_min_array[i] = host_matrix[0];
        host_max_array[i] = host_matrix[0];
    }
    cudaMemcpy(device_min_array, host_min_array, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_max_array, host_max_array, sizeof(float) * N, cudaMemcpyHostToDevice);

    get_max_min_kernel <<< blocksPerGrid, threadsPerBlock >>> (device_matrix, device_min_array, device_max_array);
    cudaDeviceSynchronize();
    cudaMemcpy(host_min_array, device_min_array, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_max_array, device_max_array, sizeof(float) * N, cudaMemcpyDeviceToHost);
    

    // 对一维数组使用无分支发散的并行规约，求其最大最小值
    float *device_min, *device_max;
    cudaMalloc((void**)&device_min, sizeof(float));
    cudaMalloc((void**)&device_max, sizeof(float));
    
    get_max_value_of_row <<< blocksPerGrid, threadsPerBlock >>> (device_max_array, device_max);
    cudaMemcpy(&host_max, device_max, sizeof(float), cudaMemcpyDeviceToHost);
    get_min_value_of_row <<< blocksPerGrid, threadsPerBlock >>> (device_min_array, device_min);
    cudaMemcpy(&host_min, device_min, sizeof(float), cudaMemcpyDeviceToHost);
    

    // 输出结果
    printf("GPU => min = %.6f, max = %.6f\n\n", host_min, host_max);


    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time elapsed on GPU: %.6f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_vector);
    cudaFree(device_vector_squared);
    cudaFree(device_matrix);
    cudaFree(device_vector_squared_sum);
    cudaFree(device_min_array);
    cudaFree(device_max_array);
    cudaFree(device_min);
    cudaFree(device_max);
    //========================= GPU end ===========================
    
    free(host_vector);
    free(host_matrix);
    free(host_min_array);
    free(host_max_array);

    return 0;
}