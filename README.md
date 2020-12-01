### 课堂作业

- [01_MatrixAddition 矩阵加法](https://github.com/sudrizzz/ParallelComputing/blob/main/01_MatrixAddition/matrix_addition.cu)
- [02_MatrixMultiplication 矩阵乘法](https://github.com/sudrizzz/ParallelComputing/blob/main/02_MatrixMultiplication/matrix_multiplication.cu)
- [03_GetMaxNumbers 获取矩阵每行最大值](https://github.com/sudrizzz/ParallelComputing/blob/main/03_GetMaxNumbers/max_number_of_rows.cu)

### 实验

#### （一）完成矩阵相乘的并行程序的实现

##### 要求

实现 2 个矩阵（1024\*1024）的相乘，M 矩阵的初始值全为本人学号的最后 1 位数字，N 矩阵的初始值全为本人学号的倒数第 2 位数字。数据类型设置为 float。需要完成 2 个版本。
参考代码见 [Lecture3B_CUDA 编程模型](https://github.com/sudrizzz/ParallelComputing/blob/main/slides/Lecture3B_CUDA编程模型.ppt)。

1. 使用 CPU 计算；
2. 使用 GPU 全局内存存放输入的矩阵 M 和 N，P 矩阵的结果存放于全局内存。

##### 代码实现

- [01_CPU 实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/matrix_multiplication_host.c)
- [02_GPU 实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/01/matrix_multiplication_device.cu)

只是做了一些微小的工作。  
Just did some tiny work.
