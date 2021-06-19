#include <stdio.h>

int lsearch(int *a, int n, int x);

int main(void) {
    int a[] = {23, -1, 4, -89, 3, 6, 467, 56, 807, 560};
    int n = sizeof a / sizeof a[0];
    int x = 56;
    int i = lsearch(a, n, x);
    printf("%d is at index %d\n", x, i);
    return 0;
}

int lsearch(int *a, int n, int x){
    int i, index;
    index = -1;
    for(i=0;i<n;++i){
        if (a[i] == x){
            index = i;
            break;
        }
    }
    return index;
}
