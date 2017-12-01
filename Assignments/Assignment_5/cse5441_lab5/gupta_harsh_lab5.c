#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "time.h"
#include <math.h>
#include "read_bmp.h"
#include <mpi.h>
#include <omp.h>
#define NPROCS 8

#define MAX_ALLOWED_CMD_ARGS  3
#define HELP_MESSAGE 	"Error in running the program. \n Usage: Provide input_image, serial_image, cuda_image as command line arguments. \n"

FILE *out_file_Serial, *out_file_Parallel, *inFile;
uint8_t *bmp_data;
uint32_t wd, ht, number_of_pixels, number_of_pixels_local;
int rank, size;

void Mpi_Omp() {
	
	// Variable Declaration
	uint8_t *new_bmp_img;
	uint32_t i, j;
	uint32_t sum1, sum2, mag;
	uint32_t threshold;
	uint32_t percent_black_cells_local = 0;
	uint32_t percent_black_cells_global = 0;
	uint32_t total_cells;
	
	struct timespec start_c, end_c; 
	double diff; 
	
	threshold   = 0;
	total_cells = wd * ht;
	
	MPI_Status status;
	
	//Allocate space for new sobel image
	new_bmp_img = (uint8_t *)malloc(number_of_pixels);
	
	uint32_t factor = ht/size;
	uint32_t s, e;
	s = rank * factor;
	// If it is the last process assign all the left rows to it
	if ( rank == size-1) {
		e = ht;
	} else {
		e = (rank+1) * factor;
	}
	

	MPI_Barrier(MPI_COMM_WORLD);
	//Allocate space for new sobel image
	number_of_pixels_local = (e - s) * wd;
	
	if (rank == 0)
	{	
		//===============================Time Calculations=======================================	
		clock_gettime(CLOCK_REALTIME,& start_c);
		//========================================================================================
	}
	
	while(percent_black_cells_global < 75)
	{	
		MPI_Barrier(MPI_COMM_WORLD);
		percent_black_cells_local = 0;
		percent_black_cells_global = 0;
		threshold += 1;
			
		#pragma omp parallel for private(mag, sum1, sum2) reduction( + : percent_black_cells_local) num_threads(128)
		for(i=s; i < e; i++)
		{
		    //#pragma omp parallel for num_threads(128)
			for(j=0; j < wd; j++)
			{
				if(i==0 || j==0 || i==ht-1 || j==wd-1) {
					new_bmp_img[i*wd+j] = 0;
					continue;
				}

				sum1 = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] \
						+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] \
						+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
						
				sum2 = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] \
						+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] \
						- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
						
				mag = sqrt(sum1 * sum1 + sum2 * sum2);
				if(mag > threshold)
				{
					new_bmp_img[ i*wd + j] = 255;
				}
				else
				{
					new_bmp_img[ i * wd + j] = 0;
					percent_black_cells_local++;
				}
			}
		}
//		#pragma omp barrier
		MPI_Allreduce(&percent_black_cells_local, &percent_black_cells_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
		percent_black_cells_global = (1.0 * percent_black_cells_global) / total_cells * 100;
	}

    //================================== Perform when rank is 0 ======================================
	if(rank == 0) {
		
		uint32_t row, column; 
		uint32_t s = 0, e = factor;
		uint32_t number_of_pixels_local_slaves;
		
	//	#pragma omp parallel for num_threads(size)
		for ( i = 1; i < size; i++) {
			
			s = i * factor;
			if ( i == size-1) {
				e = ht;
			} else {
				e = (i+1) * factor;
			}
			
			number_of_pixels_local_slaves = (e - s) * wd;
			MPI_Recv(&new_bmp_img[s*wd], number_of_pixels_local_slaves, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &status);
		}	
		//===============================Time Calculations=======================================
		clock_gettime(CLOCK_REALTIME,& end_c);  
		
		diff = ((double)end_c.tv_sec + 1.0e-9*end_c.tv_nsec) - ((double)start_c.tv_sec + 1.0e-9*start_c.tv_nsec);
		//========================================================================================
		
		printf("%s","***********************************************************************\n");
		printf("Parallel Execution \n");
		printf("Elapsed time for Sobel Operation time : %lf \n",diff);
		printf("Theshold: %d\n",threshold);
		printf("***********************************************************************\n");
		
		//Write the buffer into the bmp file
		write_bmp_file(out_file_Parallel, new_bmp_img);
		//=================================================================
		
	} else {
	    //========================== Perform if it is a slave process ==============================
		MPI_Send( &new_bmp_img[s*wd], number_of_pixels_local, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
	}
	
	free(new_bmp_img);
}

int main(int argc, char *argv[])
{
	int cmd_arg;
	
	/*First Check if no of arguments is permissible to execute*/
	if (argc > MAX_ALLOWED_CMD_ARGS)
	{
		perror(HELP_MESSAGE);
		exit(-1);
	}
	
	/*Roll over the command line arguments and obtain the values*/
	for (cmd_arg = 1; cmd_arg < argc; cmd_arg++)
	{
		/*Switch execution based on the pass of commad line arguments*/
		switch (cmd_arg)
		{
			case 1:
                inFile = fopen(argv[cmd_arg], "rb");
				break;
			case 2: 
                out_file_Parallel = fopen(argv[cmd_arg], "wb");
				break;
		}
	}
	
	//Read the binary bmp file into buffer
	bmp_data = (uint8_t *)read_bmp_file(inFile);
	
	//Get image attributes
    wd = get_image_width();
    ht = get_image_height();
	number_of_pixels = get_num_pixel();

    //MPI Initialization 
    MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
    Mpi_Omp();
    
    MPI_Finalize();
	return 0;
}