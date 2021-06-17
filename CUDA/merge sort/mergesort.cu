#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 262144
#define threadSize 1

void printArray(long A[], int size)
{
	int i;
	for (i = 0; i < size; i++)
		printf("%d ", A[i]);
	printf("\n");
}

__global__ void gpu_MergeSort(long* source, long *dest, long size) {
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N)
	{
		long start = index * size;

		long middle = start + size / 2;

		long end = start + size;
		if (end > N)
		{
			end = N;
		}

		//printf("start: %d - Middle: %d - End: %d\n", start, middle, end);

		long i = start, j = middle;

		long k = start;
		while (i < middle && j < end) {
			if (source[i] <= source[j]) {
				dest[k] = source[i];
				i++;
			}
			else {
				dest[k] = source[j];
				j++;
			}
			k++;
		}
		while (i < middle) {
			dest[k] = source[i];
			k++;
			i++;
		}
		while (j < end) {
			dest[k] = source[j];
			k++;
			j++;
		}
	}
	__syncthreads();
}


int main()
{
	float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);
        
    long a[N], b[N], *d_A, *d_B;
	
	for (size_t i = 0; i < N; i++)
	{
		a[i] = rand() % N;
		b[i] = a[i];
	}

	int size = N * sizeof(long);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	
	cudaEventRecord(total_start);
	
	cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
	
	cudaEventRecord(comp_start);
	
	long blockSize = 0;
	
	// MERGE SORT WITH GPU
	for (size_t i = 2; i <= N; i=i*2)
	{
		blockSize = N / (threadSize * i);
		//printf("block: % d - thd: %d - i: %d\n", blockSize, threadSize, i);
		gpu_MergeSort <<<blockSize, threadSize>> >(d_A, d_B, i);
		cudaDeviceSynchronize();
		cudaMemcpy(a, d_B, size, cudaMemcpyDeviceToHost);
		
		// Swap source with destination array
		long *temp = d_A;
		d_A = d_B;
		d_B = temp;
	}
	
	cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
    
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
    
  
    cudaFree(d_A);
    cudaFree(d_B);
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
    
	 printArray(a, N);
    return 0;
}


