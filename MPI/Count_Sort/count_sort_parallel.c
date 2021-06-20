#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#define N 30000
#define UPPER N*4
#define LOWER 1

void count_sort(int a[], int n, int start, int stop);
void print_array(int a[], int n);
void init_array(int array[], int n, int upper, int lower);
void display_time(double start, double end);

int main(int argc, char *argv[]) {
	
	int rank, size;
	double begin, end;
	int initial_array[N], sorted_array[N];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
    	init_array(initial_array, N, UPPER, LOWER);
    	(void) printf("Initial array: ");
        print_array(initial_array, N);
        (void) printf("Sorting began...\n\n");
	}
	
	MPI_Bcast(initial_array, N, MPI_INT, 0, MPI_COMM_WORLD);
	
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
	
	count_sort(initial_array, N, start, stop);
	
	MPI_Reduce(initial_array, sorted_array, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if(rank == 0) {
		end = MPI_Wtime();
		
		(void) printf("\nSorted array: ");
        print_array(sorted_array, N);
        display_time(begin, end);
	}
	
	MPI_Finalize();
	
	return 0;
}

/*
	parameters: matrix of numbers a[] and size n of matrix a[]
*/
void count_sort(int a[], int n, int start, int stop) {
	int i, j, count;
	int* temp = calloc(n,sizeof(int));
	
	for(i=start;i<stop;i++) {
     	count = 0;
     	for (j = 0; j < n; j++)
            if ((a[j] < a[i]) || (a[j] == a[i] &&  i< j))
                count++;
        temp[count] = a[i];
	}
	
	memcpy(a, temp, n*sizeof(int));
    free(temp);
}

void print_array(int a[], int n) {
	int i;
	
	printf("[ ");
	for(i=0;i<n;i++) {
		printf("%d ", a[i]);
	}
	printf("] \n\n");
}

void init_array(int array[], int n, int upper, int lower){
    int i;    
    for (i=0; i<n; ++i)
        array[i] = (rand() % (upper - lower + 1)) + lower;
}

void display_time(double start, double end){
    (void) printf("Time spent for sorting: %f seconds\n", (double)(end-start));
}
