#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void create_matrix(float *x, float *b, float **a, int start, int stop);
void calculate_x_matrix(float *x, float *b, float **a, int start, int stop);
void print_results(float *x, int N);
void validate_results(float *x, float *b, float **a, int N);

int main ( int argc, char *argv[] )  {

	int   i, j, N;
	float *x, *b, **a, sum, *final_x;
	char any;
	int rank, size;
	double begin, end;
	
	

	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(rank == 0) {
		/* Allocate space for matrices */
		a = (float **) malloc ( N * sizeof ( float *) );
		for ( i = 0; i < N; i++) 
			a[i] = ( float * ) malloc ( N * sizeof ( float ) );
		b = ( float * ) malloc ( N * sizeof ( float ) );
		x = ( float * ) malloc ( N * sizeof ( float ) );
	}
	
	printf("Hello -1");
	
    MPI_Bcast(a, N*N,MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, N,MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(x, N,MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	printf("Hey 0");
	
	if(rank == 0){
		begin = MPI_Wtime();
	}
	
	int chunk = N / size;
	int extra = N % size;
	int start = rank * chunk;
	int stop = start + chunk;
	
	if(rank == size - 1) {
		stop += extra;
	}
	
	
	
	/* Create floats between 0 and 1. Diagonal elents between 2 and 3. */
	create_matrix(x, b, a, start, stop); 

    /* Calulation */
	calculate_x_matrix(x, b, a, start, stop);
	
	printf("Hey 1");

    //scanf ("%c", &any);
    
    MPI_Reduce(x, final_x, N, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        
    /* Print result for debugging*/
    if(rank == 0){
    	print_results(final_x, N);	
	}
	
	MPI_Finalize();
	
	return 0;
	
		
    /* Validate  result for debugging */
//    validate_results(x, b, a, N);   
}	

void create_matrix(float *x, float *b, float **a, int start, int stop) {
	srand ( time ( NULL));
	int i, j;
	for (i = start; i < stop; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);;
	} 
}

void calculate_x_matrix(float *x, float *b, float **a, int start, int stop) {
	int i, j;
	float sum;
	#
	for (i = start; i < stop; i++) {
		sum = 0.0;
		for (j = 0; j < i; j++) {
			sum = sum + (x[j] * a[i][j]);
		}	
		x[i] = (b[i] - sum) / a[i][i];
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
