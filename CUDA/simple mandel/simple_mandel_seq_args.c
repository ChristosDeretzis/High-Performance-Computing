#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define         X_RESN  100       /* x resolution */
#define         Y_RESN  100       /* y resolution */
#define         X_MIN   -2.0
#define         X_MAX    2.0
#define         Y_MIN   -2.0
#define         Y_MAX    2.0
#define         GRADES  100



typedef struct complextype
        {
        float real, imag;
        } Compl;

int main ( int argc, char* argv[])
{

       /* Mandlebrot variables */
        int i, j, k, l;
        Compl   z, c;
        float   lengthsq, temp;
        int maxIterations;
        int res[X_RESN][Y_RESN]; 
        int gray[GRADES];


        /* Calculate and draw points */
        if (argc != 2) {
		printf ("Usage : %s <number of iterations>\n", argv[0]);
		return 1;
        }
        maxIterations = strtol(argv[1], NULL, 10);
        for (l=0; l<GRADES; l++) 
            gray[l] = (l+1)*maxIterations/GRADES;
     

        for(i=0; i < Y_RESN; i++) 
          for(j=0; j < X_RESN; j++) {
            z.real = z.imag = 0.0;
            c.real = X_MIN + j * (X_MAX - X_MIN)/X_RESN;
            c.imag = Y_MAX - i * (Y_MAX - Y_MIN)/Y_RESN;
            k = 0;

            do  {    /* iterate for pixel color */

              temp = z.real*z.real - z.imag*z.imag + c.real;
              z.imag = 2.0*z.real*z.imag + c.imag;
              z.real = temp;
              lengthsq = z.real*z.real+z.imag*z.imag;
              k++;

            } while (lengthsq < 4.0 && k < maxIterations);

           for (l=0; l<GRADES; l++)  
            if (k <= gray[l]) {
               res[i][j] = l; 
               break;
            }
        }

        printf("\n");
        for(i=0; i < Y_RESN; i++) {
          for(j=0; j < X_RESN; j++) 
            printf("\x1b[38;2;40;177;%dm%3d\x1b[0m", 101+10*res[i][j], res[i][j]);
        printf("\n");
        }
        

        /* Program Finished */
        return 0;
}
