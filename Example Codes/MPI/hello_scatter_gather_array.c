#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main( int argc, char *argv[] )
{
    int rank, size, i, *a, *tmp;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0) {
    	a = malloc (10*size*sizeof(int));
      	for (i = 0; i < 10*size; i++) {
 		a[i] = i;
                printf("%d  ", a[i]);
        }
        printf("\n");
    }
    tmp = malloc (10*sizeof(int));

    MPI_Scatter(a, 10, MPI_INT, tmp, 10, MPI_INT, 0, MPI_COMM_WORLD);

    /*if (rank == 2) {
   	for (i = 0; i < 10; i++) {
 		printf("%d  ", tmp[i]);
        }
        printf("\n");
    }*/

    for (i = 0; i < 10; i++)
    	tmp[i] = tmp[i] * rank;  

    /*if (rank == 2) {
   	for (i = 0; i < 10; i++) {
 		printf("%d  ", tmp[i]);
        }
        printf("\n");
    }*/

    MPI_Gather(tmp, 10, MPI_INT, a, 10, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
     	for (i = 0; i < 10*size; i++) 
        	printf("%d  ", a[i]);
        printf("\n");
    }

    MPI_Finalize(); 
    return 0; 
}

