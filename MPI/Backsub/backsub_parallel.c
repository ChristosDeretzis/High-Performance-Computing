#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define tag 100

void create_matrix(float *x, float *b, float *a, int n);
void calculate_x_matrix(float *x, float *b, float *a, int start, int stop, int n);
void print_results(float *x, int N);
void validate_results(float *x, float *b, float **a, int N);

int main(int argc, char *argv[]) {
	int N, i, j, size, rank;
	float *a, *b, *x, sum;
	double begin, end;
	
	a = (float *) malloc ( N * N * sizeof ( float ) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	x = ( float * ) malloc ( N * sizeof ( float ) );
	
	if(argc != 2) {
		printf("Usage: %s <matrix size>\n", argv[0]);
		exit(1);
	}
	N = strtol(argv[1], NULL, 10);
	
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank == 0) {
		create_matrix(x,b,a,N);
	}
	
	MPI_Bcast(a, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	
	begin = MPI_Wtime();
	
	int chunk = N / size;
	int extra = N % size;
	int start = rank * chunk;
	int stop = start + chunk;
	
	if(rank == (size - 1)){
		stop += extra;
	}
	
	if(rank == 0){
		calculate_x_matrix(x,b,a,start,stop,N);
		
		MPI_Send(x, N, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
	}
	
	if(rank!=0 && rank!=(size - 1)){
		MPI_Recv(x, N, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
		calculate_x_matrix(x,b,a,start,stop, N);
		MPI_Send(x, N, MPI_FLOAT, rank+1, tag, MPI_COMM_WORLD);
	}
	
	if(rank == (size - 1)){
		MPI_Recv(x, N, MPI_FLOAT, rank-1, tag, MPI_COMM_WORLD, &status);
		calculate_x_matrix(x,b,a,start,stop,N);
		
		end = MPI_Wtime() - begin;
		
		print_results(x, N);
		printf("Total time: %f\n",end);
	}
	
	MPI_Finalize();
	
	return 0;
}

void create_matrix(float *x, float *b, float *a, int n) {
	srand ( time ( NULL));
	int i, j;
	srand ( time ( NULL));
	for (i = 0; i < n; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		a[i*n + i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i*n + j] = (float)rand()/(RAND_MAX*2.0-1.0);;
	} 
}

void calculate_x_matrix(float *x, float *b, float *a, int start, int stop, int n) {
	int i, j;
	float sum;
	for (i = start; i < stop; i++) {
		sum = 0.0;
		for (j = 0; j < i; j++) {
			sum = sum + (x[j] * a[i*n +j]);
		}	
		x[i] = (b[i] - sum) / a[i*n + i];
	}
}

void print_results(float *x, int N) {
	int i;
	for (i = 0; i < N; i++) {
		printf ("%f \n", x[i]);
	}
}

void validate_results(float *x, float *b, float **a, int N) {
	int i, j;
	float sum;
	for (i = 0; i < N; i++) {
            sum = 0.0;
      		for (j = 0; j < N; j++) {
         		sum = sum + (x[j] * a[i][j]);
      			if (abs(b[i] - sum) > 0.00001) {
         			printf("%f != %f\n", sum, b[i]);
         			printf("Validation Failed...\n");
      			}
		}
	}
}
