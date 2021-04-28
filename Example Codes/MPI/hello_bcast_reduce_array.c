#include <stdio.h>
#include "mpi.h"

int main( int argc, char *argv[] )
{
    int rank, size, i, tmp[10], sumtmp[10];
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (rank == 0) 
    {	
	for (i=0; i< 10; i++)
		tmp[i] = 0;
    }
        
    MPI_Bcast(tmp, 10, MPI_INT, 0, MPI_COMM_WORLD);

    for (i=0; i<10; i++)
	tmp[i] = tmp[i]+rank+i;  
    
    MPI_Reduce(tmp, sumtmp, 10, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
    	for (i=0; i< 10; i++)
		printf("%d  ", sumtmp[i]);
	printf("\n");
    }

    MPI_Finalize(); 
    return 0; 
}

