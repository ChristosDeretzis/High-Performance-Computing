#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N 100000
#define UPPER N*4
#define LOWER 1

void count_sort(int a[], int n);
void print_array(int a[], int n);
void init_array(int array[], int n, int upper, int lower);
void display_time(clock_t start, clock_t end);

int main(void) {
	
	int array[N];

    init_array(array, N, UPPER, LOWER);

//    (void) printf("Initial array: ");
//    print_array(array, N);

    (void) printf("Sorting began...\n\n");
    double begin = clock();
    count_sort(array, N);
    double end = clock();

    display_time(begin, end);
    
//    (void) printf("\n\nSorted array: ");
//    print_array(array, N);

    return 0;
	
	return 0;
}

/*
	parameters: matrix of numbers a[] and size n of matrix a[]
*/
void count_sort(int a[], int n) {
	int i, j, count;
	int* temp = malloc(n*sizeof(int));
	
	# pragma omp parallel for private(i,j,count) shared(a)
	for(i=0;i<n;i++) {
     	count = 0;
     	for (j = 0; j < n; ++j)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
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

void display_time(clock_t start, clock_t end){
    (void) printf("Time spent for sorting: %g seconds\n", (double)(end-start) / CLOCKS_PER_SEC);
}
