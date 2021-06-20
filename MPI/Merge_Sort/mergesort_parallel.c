#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void generate_list(int * x, int n) {
   int i;
   srand (time (NULL));
   for (i = 0; i < n; i++)
     x[i] = rand() % n; 
}

void print_list(int * x, int n) {
   int i;
   for (i = 0; i < n; i++) {
      printf("%d ",x[i]);
   } 
}

void merge(int * X, int n, int * tmp) {
   int i = 0;
   int j = n/2;
   int ti = 0;

   while (i<n/2 && j<n) { /* merge */
      if (X[i] < X[j]) {
         tmp[ti] = X[i];
         ti++; i++;
      } else {
         tmp[ti] = X[j];
         ti++; j++;
      }
   }
   while (i<n/2) { /* finish up lower half */
      tmp[ti] = X[i];
      ti++; i++;
   }
   while (j<n) { /* finish up upper half */
       tmp[ti] = X[j];
       ti++; j++;
   }
   memcpy(X, tmp, n*sizeof(int));

} 

void mergesort(int * X, int n, int * tmp)
{
   if (n < 2) return;

   mergesort(X, n/2, tmp); 
   mergesort(X+(n/2), n-(n/2), tmp);
   
   merge(X, n, tmp);
}


void main(int argc, char *argv[])
{
   int n, world_rank, world_size;
   int *data;
   double start, end;
   
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
   if (argc != 2) {
		printf ("Usage : %s <list size>\n", argv[0]);
   }
   n = strtol(argv[1], NULL, 10);
   data = (int *) malloc (sizeof(int)*n);
   
   generate_list(data, n);
 
   if(world_rank == 0) {
   	   start = MPI_Wtime(); 
   }
   int size = n/world_size;
   
   int *sub_array = malloc(size * sizeof(int));
   MPI_Scatter(data, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
   
   int *tmp_array = malloc(size * sizeof(int));
   mergesort(sub_array, size, tmp_array);
   
    int *sorted = NULL;
	if(world_rank == 0) {	
		sorted = malloc(n * sizeof(int));
	}
	
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(world_rank == 0) {
		end = MPI_Wtime();
		int *other_array = malloc(n * sizeof(int));
		mergesort(sorted, n, other_array);
		
//		print_list(sorted, n);
		printf("\nTime spent for sorting: %f seconds\n", (double)(end-start));
		
		free(sorted);
		free(other_array);
	}
   free(data);
   free(sub_array);
   free(tmp_array);
   
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

}

