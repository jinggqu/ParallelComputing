![](https://visitor-badge.glitch.me/badge?page_id=sudrizzz.ParallelComputing)

# 课堂作业

1. [矩阵加法](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/01/matrix_addition.cu)
2. [矩阵乘法](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/02/matrix_multiplication.cu)
3. [获取矩阵每行最大值](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/03/max_number_of_rows.cu)
4. [一维数组卷积](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/04/convolution1d.cu)
5. [二维数组卷积](https://github.com/sudrizzz/ParallelComputing/blob/main/classwork/04/convolution2d.cu)

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

## （二）获取矩阵每一行的最大值

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

# 实验二

随机生成 8192 个随机整数，称为 R[8192]，作为列向量。R 与 R 转置的乘积除以 R 的模，然后得到矩阵 A，求出 A 所有元素的最大最小值，记为 max 与 min。

## 代码实现

[测试数据](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/02/testdata6.txt)

[代码实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/02/max_min_value_of_matrix.cu)

## 实验结果

下表计时单位为毫秒（ms）。

| 实验序号 | CPU 耗时   | GPU 耗时  |
| -------- | ---------- | --------- |
| 1        | 323.000000 | 92.157951 |
| 2        | 321.000000 | 91.480003 |
| 3        | 327.000000 | 86.834435 |

三次运行耗时平均值分别为：CPU 323.67 毫秒，GPU 90.16 毫秒，相比之下，GPU 运行速度是 CPU 的 358% 倍。

# 实验三

利用扫描方法并行创建二维和表，并完成基于和表方法的互相关系数计算。具体实验要求如下。

1. 编写 CPU 串行代码实现基于标准的归一化互相关算法的互相关系数的计算（完成）；
2. 编写 CPU 串行代码实现基于基于和表的归一化互相关算法的互相关系数的计算（完成）；
3. 完成基于纹理内存的并行扫描和表创建，并用和表算法计算以下兴趣区对应的互相关系数（未完成）；
4. 完成基于全局内存的并行扫描和表创建，并用和表算法计算以下兴趣区对应的互相关系数（完成，待优化）。

## 代码实现

[代码实现](https://github.com/sudrizzz/ParallelComputing/blob/main/experiment/03/ncc.cu)

## 实验结果

待完善

> 只是做了一些微小的工作。  
> Just did some tiny work.
