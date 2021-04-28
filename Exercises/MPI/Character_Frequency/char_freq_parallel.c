#include <stdio.h> 
#include <stdlib.h> 
#include "mpi.h"

#define N 128
#define base 0

void show_characters_frequency(int* freq);
void calculate_character_frequency(int* freq, char* buffer, long file_size);
void initialize_frequency_array(int* freq);

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int size, rank;
	int * total_freq;
	double begin, end;
	
	
	MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
		printf ("Usage : %s <file_name>\n", argv[0]);
		return 1;
        }
        
	filename = argv[1];
	pFile = fopen ( filename , "rb" );
	if (pFile==NULL) {printf ("File error\n"); return 2;}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	file_size = ftell (pFile);
	rewind (pFile);
	
	if(rank == 0){
		printf("file size is %ld\n", file_size);
		begin = MPI_Wtime();
	}
	
	 /* These initialization will be done by all processes   */
    int* freq = (int*) calloc(sizeof(int), N);
    if (freq == NULL) {printf ("Memory error\n"); return 3;}
    int chunk = file_size / size;
    int extra = file_size % size;
    int start = rank * chunk;
    int stop = start + chunk;
    if (rank == size - 1) stop += extra;

    int local_file_size = stop - start;
	
	
	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*local_file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}
	
	fseek(pFile, start, SEEK_SET);

	// copy the file into the buffer:
	fread (buffer,1,local_file_size,pFile);
	
	initialize_frequency_array(freq);
			
	calculate_character_frequency(freq, buffer, local_file_size);
	
	if (rank == 0) end = MPI_Wtime();
	
    MPI_Reduce(freq, total_freq, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0){
        show_characters_frequency(total_freq);		
		(void) printf("Time spent for counting: %g\n", (double)(end-begin));
    }

	fclose (pFile);
	free (buffer);
	
	MPI_Finalize();

	return 0;
}


void show_characters_frequency(int* freq) {
	int j;
	for ( j=0; j<N; j++){
		printf("%d = %d\n", j+base, freq[j]);
	}
}

void calculate_character_frequency(int* freq, char* buffer, long file_size) {
	int i;
	for (i=0; i<file_size; i++){
		freq[buffer[i] - base]++;
	}
}

void initialize_frequency_array(int* freq) {
	int j;
	for (j=0; j<N; j++){
		freq[j]=0;
	}
}
