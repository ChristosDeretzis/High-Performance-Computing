# include <stdlib.h>
# include <stdio.h>
# include <time.h>

void main ( int argc, char *argv[] )
  
{
  double *a, *b, *c;
  int i, j, N;
  
        if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
                exit(1);
	}
	N = strtol(argv[1], NULL, 10);

  	/*
  	Allocate the matrices.
  	*/
  	a = ( double * ) malloc ( N * N * sizeof ( double ) );
  	b = ( double * ) malloc ( N * sizeof ( double ) );
  	c = ( double * ) malloc ( N * sizeof ( double ) );
  	/*
  	Assign values to the B and C matrices.
  	*/
  	srand ( time ( NULL));

  	for ( i = 0; i < N; i++ ) 
    		for (j = 0; j < N; j++ )
	      		a[i*N+j] = ( double ) rand() / (RAND_MAX * 2.0 - 1.0);

	for ( i = 0; i < N; i++ )
	    	b[i] = ( double ) rand() / (RAND_MAX * 2.0 - 1.0);
   

  	/* computation */
        for ( i = 0; i < N; i++) {
   		 c[i] = 0.0;
    		for ( j = 0; j < N; j++ )
        		c[i] = c[i] + a[i*N+j] * b[j];
  	}
  
 
	for ( i = 0; i < N; i++ ) {
	    	for (j = 0; j < N; j++ )
	      		printf ("%1.3f ", a[i*N+j]); 
	    	printf("\t %1.3f ", b[i]);
	    	printf("\t %1.3f \n", c[i]);
       	}
 
}

