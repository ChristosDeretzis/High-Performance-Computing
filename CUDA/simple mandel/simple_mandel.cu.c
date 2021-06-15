#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define         X_RESN  50       /* x resolution */
#define         Y_RESN  50     /* y resolution */
#define         X_MIN   -2.0
#define         X_MAX    2.0
#define         Y_MIN   -2.0
#define         Y_MAX    2.0
#define         GRADES  100
#define THREADS_PER_BLOCK 4

typedef struct complextype
        {
        float real, imag;
        } Compl;
        
__device__ int getNumK(Compl c, Compl z, int max_iters);
__global__ void mandelbrot_set(int *gray, int gray_size, int *res, int x_size, int y_size, int max_iters);
void print_results(int *res, int x_size, int y_size);


int main ( int argc, char* argv[])
{

       /* Mandlebrot variables */
        int maxIterations = 1000;
     
        float total_time, comp_time;
        cudaEvent_t total_start, total_stop, comp_start, comp_stop;
        cudaEventCreate(&total_start);
        cudaEventCreate(&total_stop);
        cudaEventCreate(&comp_start);
        cudaEventCreate(&comp_stop);
       
        /*
         * Read arguments
         */
 
        int *gray = (int *)malloc(GRADES*sizeof(int));
        int *res = (int *)malloc(X_RESN*Y_RESN*sizeof(int *));
 
        /* Calculate and draw points */
         if (argc != 2) {
			printf ("Usage : %s <number of iterations>\n", argv[0]);
			return 1;
	    }
        maxIterations = strtol(argv[1], NULL, 10);
        
        for (int l=0; l<GRADES; l++) 
            gray[l] = (l+1)*maxIterations/GRADES;
 
        /*
        * Memory allocation on device
        */
        int *gray_dev, *res_dev;
        cudaMalloc((void **)&gray_dev, GRADES*sizeof(int));
        cudaMalloc((void **)&res_dev, X_RESN*Y_RESN*sizeof(int));
        
        cudaEventRecord(total_start);
 
        cudaMemcpy(gray_dev, gray, GRADES*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(res_dev, res, X_RESN*Y_RESN*sizeof(int), cudaMemcpyHostToDevice);
 
        /*
         * Create sufficient blocks 
         */
        dim3 block(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
        dim3 grid;
        grid.x = (X_RESN + block.x - 1)/block.x;
        grid.y = (Y_RESN + block.y - 1)/block.y;

        cudaEventRecord(comp_start);
	      /*
         * Kernel call
         */ 
	      mandelbrot_set<<<grid, block>>>(gray_dev, GRADES,res_dev, X_RESN, Y_RESN, maxIterations);

        cudaEventRecord(comp_stop);
        cudaEventSynchronize(comp_stop);
        cudaEventElapsedTime(&comp_time, comp_start, comp_stop);

        /*
        * Copy c from host device memory to host memory
        */
        cudaMemcpy(res, res_dev, X_RESN*Y_RESN*sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(total_stop);
        cudaEventSynchronize(total_stop);
        cudaEventElapsedTime(&total_time, total_start, total_stop);
        /*
        * Free memory on device
        */
        cudaFree(res_dev);
        cudaFree(gray_dev);
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
        print_results(res, X_RESN, Y_RESN);

        /* Program Finished */
        return 0;
}

__device__ int getNumK(Compl c, Compl z, int max_iters){
    float lengthsq, temp;
    int k = 0;
    do  {    /* iterate for pixel color */
      temp = z.real*z.real - z.imag*z.imag + c.real;
      z.imag = 2.0*z.real*z.imag + c.imag;
      z.real = temp;
      lengthsq = z.real*z.real+z.imag*z.imag;
      k++;
   } while (lengthsq < 4.0 && k < max_iters);

   return k;
}

__global__ void mandelbrot_set(int *gray, int gray_size, int *res, int x_size, int y_size, int max_iters){
    int k, l;
    Compl c, z;
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    if(true){
      z.real = z.imag = 0.0;
      c.real = X_MIN + x_index * (X_MAX - X_MIN)/X_RESN;
      c.imag = Y_MAX - y_index * (Y_MAX - Y_MIN)/Y_RESN;
      
      k = getNumK(c, z, max_iters);
      for (l=0; l<gray_size; l++)  
        if (k <= gray[l]) {
            res[y_index*y_size + x_index] = l; 
            break;
        }   
    }
     
}


void print_results(int *res, int x_size, int y_size) {
    printf("\n");
    for(int i=0; i < x_size*y_size; i++) {
      printf("\x1b[38;2;40;177;%dm%3d\x1b[0m", 101+10*res[i], res[i]);
      if((i+1) % (x_size) == 0)
        printf("\n");
      }
}
