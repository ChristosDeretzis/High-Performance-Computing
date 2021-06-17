#include <stdio.h>
#include <math.h>

#define N 20

int main(void){

    int i, j, k=0, is_prime=1;
    int primes[N];
    
    for(i=2; i<N; ++i){
        for (j=2; j<i/2+1; ++j){
            if (!(i % j) && j != i){
                is_prime = 0;
                break;
            }
        }
        if (is_prime) primes[k++] = i;
        is_prime = 1;
    }

    printf("Primes: ");
    for(i=0; i<k; ++i)
        printf("%d ", primes[i]);

    return 0;

}
