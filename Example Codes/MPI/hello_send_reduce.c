#include <stdio.h>
#include "mpi.h"

int main( int argc, char *argv[] )
{
    int rank, size, tmp, sumtmp, i;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if (rank == 0) 
    { 
    scanf("%d", &tmp);
    for (i=1; i<size; i++)
    	MPI_Send(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    else MPI_Recv(&tmp, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    tmp = tmp+rank;  
    printf( "Hello world %d from process %d of %d\n", tmp, rank, size );
    MPI_Reduce(&tmp, &sumtmp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hello total %d \n", sumtmp);
    MPI_Finalize(); 
    return 0; 
}

