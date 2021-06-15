#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define N 128
#define base 0
#define THREADS_PER_BLOCK 256

void show_characters_frequency(int *freq, int n);
void initialize_frequency_array(int *freq, int n);
__global__ void calculate_character_frequency(char *buffer, int *freq, long file_size);

int main(int argc, char *argv[]) {
    int blocks;	
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);
      
    FILE *pFile;
    long file_size;
    char * buffer;
    char *filename = "/text8.txt";
    size_t result;
    int * freq;
    
    if (argc != 2) {
      printf ("Usage : %s <file_name>\n", argv[0]);
      return 1;
    }
    
	filename = argv[1];
    pFile = fopen ( filename , "rb" );
    if (pFile==NULL) {printf ("File error\n"); return 2;}

    /* obtain file size */
    fseek (pFile , 0 , SEEK_END);
    file_size = ftell (pFile);
    rewind (pFile);
    printf("file size is %ld\n", file_size);
	
    /* allocate memory to contain the file	*/
    buffer = (char*) malloc (sizeof(char)*file_size);
    if (buffer == NULL) {printf ("Memory error\n"); return 3;}

    /* copy the file into the buffer */
    result = fread (buffer,1,file_size,pFile);
    if (result != file_size) {printf ("Reading error\n"); return 4;} 
    
    freq = (int*) malloc(sizeof(int)*N);
    if (freq == NULL) {printf ("Memory error\n"); return 5;}

	/*
	 * Memory allocation on device
	 */
    char *buff_dev;
    int *freq_dev;
    cudaMalloc((void **)&buff_dev, file_size*sizeof(char));
    cudaMalloc((void **)&freq_dev, N*sizeof(int));
    cudaMemset(freq_dev, 0, N);
	
    cudaEventRecord(total_start);

    /*
     * Copy buffer from host memory to device memory
     */
	  cudaMemcpy(buff_dev, buffer, sizeof(char)*file_size, cudaMemcpyHostToDevice);
	
   /*
    * Create sufficient blocks 
    */
    blocks = (file_size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    cudaEventRecord(comp_start);
	 /*
    * Kernel call
    */ 
	  calculate_character_frequency<<< blocks, THREADS_PER_BLOCK >>>(buff_dev, freq_dev, file_size);

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

   /*
    * Copy c from host device memory to host memory
    */
	cudaMemcpy(freq, freq_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
	
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
    
	  /*
	   * Free memory on device
     */
    cudaFree(buff_dev);
    cudaFree(freq_dev);
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

    /*
    * Display Results
    */
    show_characters_frequency(freq, N);  
        
	return 0;
}

void show_characters_frequency(int *freq, int n) {
	int j;
	for ( j=0; j<n; j++){
		printf("%d = %d\n", j+base, freq[j]);
	}
}

void initialize_frequency_array(int *freq, int n) {
	int j;
	for (j=0; j<n; j++){
		freq[j]=0;
	}
}

__global__ void calculate_character_frequency(char *buffer, int *freq, long file_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;     
    int total_threads = gridDim.x * blockDim.x;
    
    long i;
    for (i=index; i<file_size; i+=total_threads)
        atomicAdd(&(freq[buffer[i] - base]), 1);
}
