// 系统头文件
#include <stdlib.h>
#include <stdio.h>

// cuda头文件
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define N 10
#define BLOCK_SIZE 8


// GPU 上的向量加法，看作一维数组相加
__global__ void vectorAdd(float *a, float *b, float *c) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int i = bx * BLOCK_SIZE + tx;
	c[i] = a[i] + b[i];
}


// 初始化向量为随机数值
void randomInit(float* data, unsigned int size) {
	srand(1);
	for (unsigned int i = 0; i < size; i++) {
		data[i] = rand() / (float) 10000;
	}	
}


// 主机端主函数
int main(void) {
	float *aH, *bH, *cH, *aD, *bD, *cD;
	
	int mem_size = N * N * sizeof(float);

	// 在主机内存申请 A，B，C 向量的空间
	aH = (float*) malloc(mem_size);
	bH = (float*) malloc(mem_size);
	cH = (float*) malloc(mem_size);

	// 在 GPU 设备申请 A，B，C 向量的空间
	cudaMalloc((void**) &aD, mem_size);
	cudaMalloc((void**) &bD, mem_size);
	cudaMalloc((void**) &cD, mem_size);

	// 初始化主机内存的 A，B 向量
	randomInit(aH, N * N);
	randomInit(bH, N * N);

	// 拷贝主机内存的 A，B 的内容到 GPU 设备的 A，B
	cudaMemcpy(aD, aH, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(bD, bH, mem_size, cudaMemcpyHostToDevice);

	// GPU 内核函数的维度参数
	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid((N * N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

	// 执行 GPU 内核函数
	vectorAdd <<<dimGrid, dimBlock >>> (aD, bD, cD);

	// 从 GPU 设备复制结果向量 C 到主机内存的 C
	cudaMemcpy(cH, cD, mem_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%.2f\t", cH[i * N + j]);
		}
		printf("\n");
	}

	free(aH);
	free(bH);
	free(cH);
	cudaFree(aD);
	cudaFree(bD);
	cudaFree(cD);
	system("pause");
}
