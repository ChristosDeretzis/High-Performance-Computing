#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void Get_dims(int* m_p, int* n_p);
void Read_matrix(char prompt[], double A[], int m, int n, int upper, int lower);
void Read_vector(char prompt [], double x[], int n, int upper, int lower);
void Print_matrix(char title[], double A[], int m, int n);
void Print_vector(char title[], double y[], int m);
void Mat_vect_mult(double A[], double x[], double y[], int m, int n);

/*-------------------------------------------------------------------*/
int main(void) {
   double* A = NULL;
   double* x = NULL;
   double* y = NULL;
   int m, n;

   Get_dims(&m, &n);
   A = malloc(m*n*sizeof(double));
   x = malloc(n*sizeof(double));
   y = malloc(m*sizeof(double));
   if (A == NULL || x == NULL || y == NULL) {
      fprintf(stderr, "Can't allocate storage\n");
      exit(-1);
   }
   Read_matrix("A", A, m, n, 12, 1);

//   Print_matrix("A", A, m, n);

   Read_vector("x", x, n, 12, 1);

//   Print_vector("x", x, m);
   double start = clock();
   Mat_vect_mult(A, x, y, m, n);
   double end = clock();
   
   Print_vector("y", y, m);
   printf("Time spent for sorting: %g seconds\n", (double)(end-start) / CLOCKS_PER_SEC);
   
   free(A);
   free(x);
   free(y);
   return 0;
}  /* main */



void Get_dims(
              int*  m_p  /* out */, 
              int*  n_p  /* out */) {
   printf("Enter the number of rows\n");
   scanf("%d", m_p);
   printf("Enter the number of columns\n");
   scanf("%d", n_p);

   if (*m_p <= 0 || *n_p <= 0) {
      fprintf(stderr, "m and n must be positive\n");
      exit(-1);
   }
}  /* Get_dims */


void Read_matrix(
                 char    prompt[]  /* in  */, 
                 double  A[]       /* out */, 
                 int     m         /* in  */, 
                 int     n         /* in  */,
				 int     upper,
				 int     lower) {
   int i, j;
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
        A[i*n+j] = (rand() % (upper - lower + 1)) + lower;
}  /* Read_matrix */


void Read_vector(
                 char    prompt[]  , 
                 double  x[]       , 
                 int     n         ,
				 int upper, int lower) {
   int i;
   
   for (i = 0; i < n; i++)
       x[i] = (rand() % (upper - lower + 1)) + lower;
}  /* Read_vector */


void Print_matrix(
                  char    title[]  /* in */,
                  double  A[]      /* in */, 
                  int     m        /* in */, 
                  int     n        /* in */) {
   int i, j;

   printf("\nThe matrix %s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%f ", A[i*n+j]);
      printf("\n");
   }
}  /* Print_matrix */


void Print_vector(
                  char    title[]  /* in */, 
                  double  y[]      /* in */, 
                  int     m        /* in */) {
   int i;

   printf("\nThe vector %s\n", title);
   for (i = 0; i < m; i++)
      printf("%f ", y[i]);
   printf("\n");
}  /* Print_vector */



void Mat_vect_mult(
                   double  A[]  /* in  */, 
                   double  x[]  /* in  */, 
                   double  y[]  /* out */,
                   int     m    /* in  */, 
                   int     n    /* in  */) {
   int i, j;

   for (i = 0; i < m; i++) {
      y[i] = 0.0;
      for (j = 0; j < n; j++)
         y[i] += A[i*n+j]*x[j];
   }
}  /* Mat_vect_mult */
