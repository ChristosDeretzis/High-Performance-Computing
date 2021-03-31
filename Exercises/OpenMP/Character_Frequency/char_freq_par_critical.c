#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h>
#define N 128
#define base 0

void show_characters_frequency(int freq[]);
void calculate_character_frequency(int freq[], char* buffer, long file_size);
void initialize_frequency_array(int freq[]);

int main (int argc, char *argv[]) {
	
	FILE *pFile;
	long file_size;
	char * buffer;
	char * filename;
	size_t result;
	int i, j, freq[N];

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
	printf("file size is %ld\n", file_size);
	
	// allocate memory to contain the file:
	buffer = (char*) malloc (sizeof(char)*file_size);
	if (buffer == NULL) {printf ("Memory error\n"); return 3;}

	// copy the file into the buffer:
	result = fread (buffer,1,file_size,pFile);
	if (result != file_size) {printf ("Reading error\n"); return 4;} 
	
	initialize_frequency_array(freq);
			
	start = omp_get_wtime();	
	calculate_character_frequency(freq, buffer, file_size);
	end = omp_get_wtime();
	
	printf("Time for counting: %g", (end-start));
	
	show_characters_frequency(freq);	

	fclose (pFile);
	free (buffer);

	return 0;
}


void show_characters_frequency(int freq[]) {
	int j;
	for ( j=0; j<N; j++){
		printf("%d = %d\n", j+base, freq[j]);
	}
}

void calculate_character_frequency(int freq[], char* buffer, long file_size) {
	int i;
	#pragma omp parallel for private(i) shared(freq, buffer, file_size)
	for (i=0; i<file_size; i++){
		#pragma omp critical
		freq[buffer[i] - base]++;
	}
}

void initialize_frequency_array(int freq[]) {
	int j;
	for (j=0; j<N; j++){
		freq[j]=0;
	}
}
