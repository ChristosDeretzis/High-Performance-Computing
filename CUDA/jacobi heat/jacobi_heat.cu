#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define maxsize 20
#define iterations 500
#define row 7
#define col 7
#define start 100
#define accuracy 50

#define THREADS_PER_BLOCK 4

__global__ void gpu_Heat (double *table_1, double *table_2, double *residual,int N) {

	int sizey = N;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
  printf("%d, %d", i, j);
	double diff=0.0;
 	if( i < N-1 && j < N-1 && i > 0 && j > 0) {
  		table_2[i*sizey+j]= 0.25 *
								 (table_1[ i*sizey     + (j-1) ]+  // left
					           table_1[ i*sizey     + (j+1) ]+  // right
				              table_1[ (i-1)*sizey + j     ]+  // top
				              table_1[ (i+1)*sizey + j     ]); // bottom
      diff = table_2[i*sizey+j] - table_1[i*sizey + j];
      residual[i*sizey+j] = diff * diff;
	}
}

__global__ void gpu_HeatReduction (double *res, double *result) {

	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int index= blockIdx.x*blockDim.x+ threadIdx.x;

	sdata[tid] = res[index];
	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
		sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		int blockIndex = blockIdx.x;

		result[blockIndex] = sdata[tid];
	}

}

int main(int argc, char *argv[]) {
    int blocks, i, j, k;	
    float total_time, comp_time;
    cudaEvent_t total_start, total_stop, comp_start, comp_stop;
    cudaEventCreate(&total_start);
  	cudaEventCreate(&total_stop);
  	cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_stop);

    double *table_1 = (double *)malloc(maxsize*maxsize*sizeof(double));
    double *table_2 = (double *)malloc(maxsize*maxsize*sizeof(double));
    double *res = (double*)malloc(sizeof(double)*maxsize*maxsize);
	  double *result = (double*)malloc(sizeof(double)*maxsize);

    for(i=0;i<maxsize;i++)
      for(j=0;j<maxsize;j++)
      {
        table_1[i*maxsize+j]=0;
        table_2[i*maxsize+j]=0;
      }

    /*
     * Memory allocation on device
     */
    double *table_1_dev, *table_2_dev, *res_dev, *result_dev;
    cudaMalloc((void **)&table_1_dev, maxsize*maxsize*sizeof(double));
    cudaMalloc((void **)&table_2_dev, maxsize*maxsize*sizeof(double));
    cudaMalloc((void **)&res_dev, maxsize*maxsize*sizeof(double));
    cudaMalloc((void **)&result_dev, maxsize*sizeof(double));
        
    cudaEventRecord(total_start);
 
    cudaMemcpy(table_1_dev, table_1, maxsize*maxsize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(table_2_dev, table_2, maxsize*maxsize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(res_dev, res, maxsize*maxsize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result, maxsize*sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(comp_start);

    dim3 block(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
    dim3 grid;
    grid.x = (maxsize + block.x - 1)/block.x;
    grid.y = (maxsize + block.y - 1)/block.y;

    for(k = 0; k < iterations; k++) {
        table_1_dev[row*maxsize + col] = start;

        gpu_Heat<<<grid, block>>>(table_1_dev, table_2_dev, res_dev, maxsize);
        cudaThreadSynchronize();  

        cudaMemcpy( res, res_dev, sizeof(double)*(maxsize*maxsize), cudaMemcpyDeviceToHost);

        gpu_HeatReduction<<<maxsize,maxsize,maxsize*sizeof(double)>>>(res_dev, result_dev);
	      cudaThreadSynchronize();

        cudaMemcpy( result, result_dev, sizeof(double)*maxsize, cudaMemcpyDeviceToHost);

        double * tmp = table_1_dev;
        table_1_dev = table_2_dev;
        table_2_dev = tmp;

        double sum =0.0, residual;
        for(int i=0;i<maxsize;i++) {	
            sum += result[i]; 	
        }
        residual = sum; 
	
       if (residual < accuracy){
           break;
       }


    }

    cudaEventRecord(comp_stop);
    cudaEventSynchronize(comp_stop);
    cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

    cudaMemcpy( table_2, table_2_dev, sizeof(double)*maxsize*maxsize, cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);
     
   /*
    * Free memory on device
    */
    cudaFree(table_1_dev);
    cudaFree(table_2_dev);
    cudaFree(res_dev);
    cudaFree(result_dev);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    printf("Total time (ms): %f\n", total_time);
    printf("Kernel time (ms): %f\n", comp_time);
    printf("Data transfer time (ms): %f\n", total_time-comp_time);

    return 0;
    
}
