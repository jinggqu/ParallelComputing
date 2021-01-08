#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1024

float standard_ncc(float* mat_f, float* mat_g, int u, int v, int wx, int wy);
float sum_table_ncc(float* mat_f, float* mat_g, int u, int v, int wx, int wy);
float get_sum_of_products(float* mat_a, float* mat_b, float* st, int u, int v);
__global__ void get_mat_sum(float* mat_a, float* mat_b, float* sum, int u, int v, int wx, int wy);

int main()
{
    int count = N * N;
    int roi1_u = 51;
    int roi1_v = 51;
    //int roi2_u = 81;
    //int roi2_v = 91;
    int roi1_wx = 900;
    int roi1_wy = 900;
    //int roi2_wx = 800;
    //int roi2_wy = 800;

    float* mat_f, * mat_g;
    mat_f = (float*)malloc(sizeof(float) * count);
    mat_g = (float*)malloc(sizeof(float) * count);
    if (mat_f == NULL || mat_g == NULL) return -1;

    for (int i = 0; i < count; i++) {
        mat_f[i] = 3.0;
        mat_g[i] = 2.0;
    }

    //========================= CPU start =========================

    // 记录程序开始运行的时间
    double start_time, end_time;
    start_time = (double)clock();

    float ncc = standard_ncc(mat_f, mat_g, roi1_u, roi1_v, roi1_wx, roi1_wy);
    printf("standard_ncc_on_CPU\t-> %.6f\n", ncc);
    end_time = (double)clock();
    printf("time elapsed on CPU\t-> %.6f ms\n\n", end_time - start_time);
    start_time = end_time;

    ncc = sum_table_ncc(mat_f, mat_g, roi1_u, roi1_v, roi1_wx, roi1_wy);
    printf("sum_table_ncc_on_CPU\t-> %.6f\n", ncc);

    // 输出程序运行花费的时间
    end_time = (double)clock();
    printf("time elapsed on CPU\t-> %.6f ms\n\n", end_time - start_time);

    //========================= CPU end ===========================


    //========================= GPU start =========================
    // 通过标准方法并行计算 NCC
    // 记录程序开始运行的时间
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 threads_per_block(N);
    dim3 number_of_blocks(N);

    float* device_mat_f, * device_mat_g;
    cudaMalloc((void**)&device_mat_f, sizeof(float) * count);
    cudaMalloc((void**)&device_mat_g, sizeof(float) * count);
    cudaMemcpy(device_mat_f, mat_f, sizeof(float) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat_g, mat_g, sizeof(float) * count, cudaMemcpyHostToDevice);

    float* device_fg_sum, * device_f2_sum, * device_g2_sum;
    float fg_sum = 0, f2_sum = 0, g2_sum = 0;
    cudaMalloc((void**)&device_fg_sum, sizeof(float));
    cudaMalloc((void**)&device_f2_sum, sizeof(float));
    cudaMalloc((void**)&device_g2_sum, sizeof(float));
    cudaMemcpy(device_fg_sum, &fg_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_f2_sum, &f2_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_g2_sum, &g2_sum, sizeof(float), cudaMemcpyHostToDevice);

    get_mat_sum << < number_of_blocks, threads_per_block >> >
        (device_mat_f, device_mat_g, device_fg_sum, roi1_u, roi1_v, roi1_wx, roi1_wy);
    get_mat_sum << < number_of_blocks, threads_per_block >> >
        (device_mat_f, device_mat_f, device_f2_sum, roi1_u, roi1_v, roi1_wx, roi1_wy);
    get_mat_sum << < number_of_blocks, threads_per_block >> >
        (device_mat_g, device_mat_g, device_g2_sum, roi1_u, roi1_v, roi1_wx, roi1_wy);

    cudaDeviceSynchronize();
    cudaMemcpy(&fg_sum, device_fg_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&f2_sum, device_f2_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&g2_sum, device_g2_sum, sizeof(float), cudaMemcpyDeviceToHost);

    ncc = fg_sum / sqrtf(f2_sum * g2_sum);
    printf("standard_ncc_on_GPU\t-> %.6f\n", ncc);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("time elapsed on GPU\t-> %.6f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //========================= GPU end ===========================

    free(mat_f);
    free(mat_g);
    cudaFree(device_mat_f);
    cudaFree(device_mat_g);
    cudaFree(device_fg_sum);
    cudaFree(device_f2_sum);
    cudaFree(device_g2_sum);
    return 0;
}


// 标准方法计算 NCC（CPU）
float standard_ncc(float* mat_f, float* mat_g, int u, int v, int wx, int wy) {
    float product = 0, f_squared_sum = 0, g_squared_sum = 0;

    for (int i = u; i < u + wx - 1 && i < N; i++) {
        for (int j = v; j < v + wy - 1 && j < N; j++) {
            product += mat_f[i * N + j] * mat_g[i * N + j];
            f_squared_sum += powf(mat_f[i * N + j], 2);
            g_squared_sum += powf(mat_g[i * N + j], 2);
        }
    }
    return product / sqrtf((float)f_squared_sum * (float)g_squared_sum);
}


// 和表方法计算 NCC（CPU）
float sum_table_ncc(float* mat_f, float* mat_g, int u, int v, int wx, int wy) {
    float* st_f_squared, * st_g_squared, * st_f_g;
    st_f_squared = (float*)malloc(sizeof(int) * N * N);
    st_g_squared = (float*)malloc(sizeof(int) * N * N);
    st_f_g = (float*)malloc(sizeof(int) * N * N);
    if (st_f_squared == NULL || st_g_squared == NULL || st_f_g == NULL)
        return -1;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            st_f_squared[i * N + j] = get_sum_of_products(mat_f, mat_f, st_f_squared, i, j);
            st_g_squared[i * N + j] = get_sum_of_products(mat_g, mat_g, st_g_squared, i, j);
            st_f_g[i * N + j] = get_sum_of_products(mat_f, mat_g, st_f_g, i, j);
        }
    }

    float product = 0.0, f_squared_sum = 0.0, g_squared_sum = 0.0;
    product = st_f_g[(u + wx - 1) * N + v + wy - 1]
        - st_f_g[(u - 1) * N + v + wy - 1]
        - st_f_g[(u + wx - 1) * N + v - 1]
        + st_f_g[(u - 1) * N + v - 1];
    f_squared_sum = st_f_squared[(u + wx - 1) * N + v + wy - 1]
        - st_f_squared[(u - 1) * N + v + wy - 1]
        - st_f_squared[(u + wx - 1) * N + v - 1]
        + st_f_squared[(u - 1) * N + v - 1];
    g_squared_sum = st_g_squared[(u + wx - 1) * N + v + wy - 1]
        - st_g_squared[(u - 1) * N + v + wy - 1]
        - st_g_squared[(u + wx - 1) * N + v - 1]
        + st_g_squared[(u - 1) * N + v - 1];

    free(st_f_squared);
    free(st_g_squared);
    free(st_f_g);
    return product / sqrtf((float)f_squared_sum * (float)g_squared_sum);
}


// 求两个矩阵的乘积（CPU）
float get_sum_of_products(float* mat_a, float* mat_b, float* st, int u, int v) {
    if (u == 0 || v == 0)
        return 0;
    else
        return mat_a[u * N + v] * mat_b[u * N + v] + st[(u - 1) * N + v]
        + st[u * N + v - 1] - st[(u - 1) * N + v - 1];
}


// 矩阵元素相乘并将结果求和（无并行规约）
__global__ void get_mat_sum(float* mat_a, float* mat_b, float* sum, int u, int v, int wx, int wy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= u * N + v && idx <= wx * N + wy)
        atomicAdd(sum, mat_a[idx] * mat_b[idx]);
}