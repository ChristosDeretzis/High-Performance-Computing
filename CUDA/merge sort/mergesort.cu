#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

__global__ void MergeSort(int *nums, int *temp, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 2; i < 2 * n; i *= 2) {
        int len = i;
        if (n - tid < len) len = n - tid;
        if (tid % i == 0) {
            int *seqA = &nums[tid], lenA = i / 2, j = 0;
            int *seqB = &nums[tid + lenA], lenB = len - lenA, k = 0;
            int p = tid;
            while (j < lenA && k < lenB) {
                if (seqA[j] < seqB[k]) {
                    temp[p] = seqA[j];
                    p++;
                    j++;
                } else {
                    temp[p] = seqB[k];
                    p++;
                    k++;
                }
            }
            while (j < lenA)
                temp[p] = seqA[j];
                p++;
                j++;
            while (k < lenB)
                temp[p] = seqB[k];
                p++;
                k++;
            for (int j = tid; j < tid + len; j++)
                nums[j] = temp[j];
        }
        __syncthreads();
    }
}

int main() {
    float total_time, comp_time;
        cudaEvent_t total_start, total_stop, comp_start, comp_stop;
        cudaEventCreate(&total_start);
        cudaEventCreate(&total_stop);
        cudaEventCreate(&comp_start);
        cudaEventCreate(&comp_stop);
    
    int size = 100;
    int *nums = (int*)malloc(sizeof(int) * size);
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        nums[i] = rand() % 3000;
    }

    int *dNums;
    cudaMalloc((void**)&dNums, sizeof(int) * size);
    int *dTemp;
    cudaMalloc((void**)&dTemp, sizeof(int) * size);

    cudaEventRecord(total_start);

    cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(10);
    dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x);

    cudaEventRecord(comp_start);

    MergeSort<<<blockNum, threadPerBlock>>>(dNums, dTemp, size);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

     for (int i = 0; i < size; ++i) {
         printf("%d ", nums[i]);
     }
     printf("\n");

    free(nums);
    cudaFree(dNums);
    cudaFree(dTemp);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    /*
    /*
     * GPU timing
     */
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);

    printf("Number of numbers: %d\n", size);
}
