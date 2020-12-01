#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

int main()
{

    // 记录程序开始运行的时间
    double startTime, endTime;
    startTime = (double)clock();

    // 分配空间，matrixP 存放结果
    int memSize = SIZE * SIZE * sizeof(float);
    float *matrixM = (float *)malloc(memSize);
    float *matrixN = (float *)malloc(memSize);
    float *matrixP = (float *)malloc(memSize);

    for (int i = 0; i < SIZE * SIZE; ++i)
    {
        matrixM[i] = 3;
        matrixN[i] = 2;
    }

    float sum, a, b;
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            sum = 0;
            // 将 M 的第 k 行与 N 的第 k 列对应相乘再相加
            for (int k = 0; k < SIZE; ++k)
            {
                a = matrixM[i * SIZE + k];
                b = matrixN[k * SIZE + j];
                sum += a * b;
            }
            // 存放一行与一列相乘的结果
            matrixP[i * SIZE + j] = sum;
        }
    }

    // 输出结果
    // for (int i = 0; i < SIZE * SIZE; ++i) {
    //    printf("%.f\t", matrixP[i]);
    // }

    // 输出程序运行花费的时间
    endTime = (double)clock();
    printf("Time elapsed: %.2f ms\n", endTime - startTime);

    free(matrixM);
    free(matrixN);
    free(matrixP);

    return 0;
}
