#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define N 200000
#define UPPER N*4
#define LOWER 1
#define THREADS_PER_BLOCK 32

/*
 * Main
 */
 
 __global__ void count_sort(int *array, int *sorted_array, int n);
 void print_array(int *a, int n);
 void init_array(int *array, int n, int upper, int lower);

int main(int argc, char *argv[])
{	
        int blocks;	
        int threads_per_block;
     
        float total_time, comp_time;
        cudaEvent_t total_start, total_stop, comp_start, comp_stop;
        cudaEventCreate(&total_start);
        cudaEventCreate(&total_stop);
        cudaEventCreate(&comp_start);
        cudaEventCreate(&comp_stop);
       
        /*
         * Read arguments
         */
    
        threads_per_block = 16;

        /*
        * Memory allocation on host 
        */
        int *array = (int *)malloc(N*sizeof(int));
        int *sorted_array = (int *)malloc(N*sizeof(int));
        
        init_array(array, N, UPPER, LOWER);
        print_array(array, N);

        /*
        * Memory allocation on device
        */
        int *array_dev, *sorted_array_dev;
        cudaMalloc((void **)&array_dev, N*sizeof(int));
        cudaMalloc((void **)&sorted_array_dev, N*sizeof(int));
        
        cudaEventRecord(total_start);

        /*
        * Copy a, b from host memory to device memory
        */
        cudaMemcpy(array_dev, array, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(sorted_array_dev, sorted_array, N*sizeof(int), cudaMemcpyHostToDevice);
        
        /*
         * Create sufficient blocks 
         */
        blocks = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

        cudaEventRecord(comp_start);
	      /*
         * Kernel call
         */ 
	      count_sort<<< blocks, THREADS_PER_BLOCK >>>(array_dev, sorted_array_dev, N);

        cudaEventRecord(comp_stop);
        cudaEventSynchronize(comp_stop);
        cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

        /*
        * Copy c from host device memory to host memory
        */
        cudaMemcpy(sorted_array, sorted_array_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(total_stop);
        cudaEventSynchronize(total_stop);
        cudaEventElapsedTime(&total_time, total_start, total_stop);
        /*
        * Free memory on device
        */
        cudaFree(sorted_array_dev);
        cudaFree(array_dev);
        cudaEventDestroy(comp_start);
        cudaEventDestroy(comp_stop);
        cudaEventDestroy(total_start);
        cudaEventDestroy(total_stop);

        /*
        /*
         * GPU timing
         */
        printf("N: %d, blocks: %d, total_threads: %d\n", N, blocks, threads_per_block*blocks);
        printf("Total time (ms): %f\n", total_time);
        printf("Kernel time (ms): %f\n", comp_time);
        printf("Data transfer time (ms): %f\n", total_time-comp_time);   
        print_array(sorted_array, N);

        
	return 0;
}

/*
 * Kernel function
 */

__global__ void count_sort(int *array, int *sorted_array, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    int j;

    if(index < n) {
      for (j = 0; j < n; ++j)
          if (array[j] < array[index])
              ++count;
          else if (array[j] == array[index] && j < index)
              ++count;
      sorted_array[count] = array[index];
    }
}

void print_array(int *a, int n) {
	int i;
	
	printf("[ ");
	for(i=0;i<n;i++) {
		printf("%d ", a[i]);
	}
	printf("] \n\n");
}

void init_array(int *array, int n, int upper, int lower){
    int i;    
    for (i=0; i<n; ++i)
        array[i] = (rand() % (upper - lower + 1)) + lower;
}
