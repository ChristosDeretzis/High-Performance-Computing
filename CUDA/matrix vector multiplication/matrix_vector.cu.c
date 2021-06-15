#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 2

__global__ void mvn(int n, float *a, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d \n", index);
    if (index < n) {
          float sum = 0.0;
          for (int j = 0; j < n; j++)
            sum = sum + a[index*n+j]*x[j];
          y[index] = sum;
    }
}

int main ( int argc, char *argv[] )
{
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);
 
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int i, j, blocks;
    
    if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);

  	/*
  	Allocate the matrices.
  	*/
  	a = ( float * ) malloc ( N * N * sizeof ( float ) );
  	b = ( float * ) malloc ( N * sizeof ( float ) );
  	c = ( float * ) malloc ( N * sizeof ( float ) );
 
    cudaMalloc(&d_a, N*N*sizeof(float)); 
    cudaMalloc(&d_b, N*sizeof(float)); 
    cudaMalloc(&d_c, N*sizeof(float));
  	/*
  	Assign values to the B and C matrices.
  	*/
  	srand ( time ( NULL));

  	for ( i = 0; i < N; i++ ) 
    		for (j = 0; j < N; j++ )
	      		a[i*N+j] = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);

    for ( i = 0; i < N; i++ )
          b[i] = ( float ) rand() / (RAND_MAX * 2.0 - 1.0);
 
    cudaEventRecord(total_start);
 
    /*
     * Copy buffer from host memory to device memory
     */
	  cudaMemcpy(d_a, a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	  cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);
 
	
   /*
    * Create sufficient blocks 
    */
    blocks = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    cudaEventRecord(comp_start);
	 /*
    * Kernel call
    */ 
	  mvn<<<blocks, THREADS_PER_BLOCK>>>(N, d_a, d_b, d_c);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);
    
    /*
    * Copy c from host device memory to host memory
    */
	  cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
	
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
	  /*
	   * Free memory on device
     */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
       
    /*
    * GPU timing
    */
 
    printf("N: %d, blocks: %d, total_threads: %d\n", N, blocks, THREADS_PER_BLOCK*blocks);
    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);
  
	for ( i = 0; i < N; i++ ) {
	    	for (j = 0; j < N; j++ )
	      		printf ("%1.3f ", a[i*N+j]); 
	    	printf("\t %1.3f ", b[i]);
	    	printf("\t %1.3f \n", c[i]);
  }
  
  return 0;
 
}
