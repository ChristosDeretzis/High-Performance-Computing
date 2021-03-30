#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void create_matrix(float *x, float *b, float **a, int N);
void calculate_x_matrix(float *x, float *b, float **a, int N);
void print_results(float *x, int N);
void validate_results(float *x, float *b, float **a, int N);

void main ( int argc, char *argv[] )  {

int   i, j, N;
float *x, *b, **a, sum;
char any;

	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);

	/* Allocate space for matrices */
	a = (float **) malloc ( N * sizeof ( float *) );
	for ( i = 0; i < N; i++) 
		a[i] = ( float * ) malloc ( N * sizeof ( float ) );
	b = ( float * ) malloc ( N * sizeof ( float ) );
	x = ( float * ) malloc ( N * sizeof ( float ) );

	/* Create floats between 0 and 1. Diagonal elents between 2 and 3. */
	create_matrix(x, b, a, N); 

    /* Calulation */
	calculate_x_matrix(x, b, a, N);

    //scanf ("%c", &any);
        
    /* Print result for debugging*/
	print_results(x, N);
		
    /* Validate  result for debugging */
//    validate_results(x, b, a, N);   
}	

void create_matrix(float *x, float *b, float **a, int N) {
	srand ( time ( NULL));
	int i, j;
	for (i = 0; i < N; i++) {
		x[i] = 0.0;
		b[i] = (float)rand()/(RAND_MAX*2.0-1.0);
		a[i][i] = 2.0+(float)rand()/(RAND_MAX*2.0-1.0);
		for (j = 0; j < i; j++) 
			a[i][j] = (float)rand()/(RAND_MAX*2.0-1.0);;
	} 
}

void calculate_x_matrix(float *x, float *b, float **a, int N) {
	int i, j;
	float sum;
	for (i = 0; i < N; i++) {
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
