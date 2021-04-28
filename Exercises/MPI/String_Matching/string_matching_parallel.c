#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include "mpi.h"

void initialize_match(int match[], long match_size);
long find_total_matches(int match[], char* pattern, char* buffer, long match_size, long pattern_size);
void show_total_matches(int match[], long match_size, long total_matches);

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size, match_size, local_match_size, pattern_size, local_total_matches, total_matches;
	char * buffer;
	char * filename, *pattern;
	size_t result;
	int i, j, *match;
	double begin, end;
	int size, rank;
	
	MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
		printf ("Usage : %s <file_name> <string>\n", argv[0]);
		return 1;
    }
	filename = argv[1];
	pattern = argv[2];
	
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
	fread (buffer,1,file_size,pFile);
	
	pattern_size = strlen(pattern);
	local_match_size = local_file_size - pattern_size + 1;
	match_size = file_size - pattern_size + 1;
	
	match = (int *) malloc (sizeof(int)*local_match_size);
	if (match == NULL) {printf ("Malloc error\n"); return 5;}
	
	initialize_match(match, local_match_size);
	
	local_total_matches = find_total_matches(match, pattern, buffer, local_match_size, pattern_size);
	printf("Matches: %ld\n", local_total_matches);
	
	if (rank == 0) end = MPI_Wtime();
	
	MPI_Reduce(&local_total_matches, &total_matches, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		printf("\nTotal matches = %ld\n", total_matches);
		printf("Time spent for counting: %g\n", (double)(end-begin));
	}
    

	fclose (pFile);
	free (buffer);
	free (match);
	
	MPI_Finalize();

	return 0;
}

void initialize_match(int match[], long match_size) {
	int j;
	for (j = 0; j < match_size; j++){
		match[j]=0;
	}
}

long find_total_matches(int match[], char* pattern, char* buffer, long match_size, long pattern_size) {
	int i, j;
	long total_matches = 0;
	
	for (i = 0; i < match_size; i++) {
      		for (j = 0; j < pattern_size; j++){
      			if (buffer[i + j] != pattern[j])
      				break;
			}
	      	if (j == pattern_size) {
	         		match[j] = 1;
	         		total_matches++;
	        }		
        }
    return total_matches;
}

void show_total_matches(int match[], long match_size, long total_matches) {
	int j;
	for (j = 0; j < match_size; j++){
		printf("%d", match[j]);
	}	
    printf("\nTotal matches = %ld\n", total_matches);
}
