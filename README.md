# 课堂作业

1. [矩阵加法](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/01/matrix_addition.cu)
2. [矩阵乘法](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/02/matrix_multiplication.cu)
3. [获取矩阵每行最大值](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/03/max_number_of_rows.cu)

# 实验一

## （一）完成矩阵相乘的并行程序的实现

### 要求

实现 2 个矩阵（1024\*1024）的相乘，M 矩阵的初始值全为本人学号的最后 1 位数字，N 矩阵的初始值全为本人学号的倒数第 2 位数字。数据类型设置为 float。需要完成 2 个版本。
参考代码见 Lecture3B_CUDA 编程模型.ppt

1. 使用 CPU 计算；
2. 使用 GPU 全局内存存放输入的矩阵 M 和 N，P 矩阵的结果存放于全局内存。

### 代码实现

1. [CPU 实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/matrix_multiplication_host.c)
2. [GPU 实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/matrix_multiplication_device.cu)

## （二）完成矩阵相乘的并行程序的实现

完成一个尺寸 512\*512 的二维数组的每一行最大值的并行程序实现。矩阵的初始值全为本人学号的最后 1 位数字。数据类型设置为 float。需要完成 4 个版本。
参考代码见 Lecture4\_性能优化-并行归约.ppt

1. 不使用共享内存，只使用全局内存，采用具有分支发散的并行归约；
2. 不使用共享内存，只使用全局内存，采用无分支发散的并行归约；
3. 使用共享内存，采用具有分支发散的并行归约；
4. 使用共享内存，采用无分支发散的并行归约。

### 代码实现

1. [全局内存分支发散](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/reduction.cu)
2. [全局内存无分支发散](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/none_reduction.cu)
3. [共享内存分支发散](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/reduction_shared.cu)
4. [共享内存无分支发散](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/none_reduction_shared.cu)

> 只是做了一些微小的工作。  
> Just did some tiny work.
