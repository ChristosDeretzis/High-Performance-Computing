#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 50000
#define UPPER N*4
#define LOWER 1

void count_sort(int a[], int n);
void print_array(int a[], int n);
void init_array(int array[], int n, int upper, int lower);
void display_time(clock_t start, clock_t end);

int main(int argc, char* argv[]) {

    int thread_count, n;
    if(argc != 3){
    	printf("Usage: <thread_count> <n>");
    }
    thread_count = strtoll(argv[1],NULL,10);
    n = strtoll(argv[2],NULL,10);
	
    int array[n];

    init_array(array, n, UPPER, LOWER);

//    (void) printf("Initial array: ");
//    print_array(array, N);

    (void) printf("Sorting began...\n\n");
    double begin = clock();
    count_sort(array, n);
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
	
	for(i=0;i<n;i++) {
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

void display_time(clock_t start, clock_t end){
    (void) printf("Time spent for sorting: %g seconds\n", (double)(end-start) / CLOCKS_PER_SEC);
}
