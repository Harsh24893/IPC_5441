#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "time.h"
#include <math.h>
#define NS_PER_US 1000

#define MAX_ALLOWED_CMD_ARGS  4
#define HELP_MESSAGE 	"Error in running the program. \n Usage: Provide input_image, serial_image, cuda_image as command line arguments. \n"

extern "C" {
#include "read_bmp.h"
}

FILE *out_file_Serial, *out_file_Parallel, *inFile;
uint8_t *bmp_data;
uint32_t wd, ht, number_of_pixels;

__global__ void sobelEdgeDetect( uint8_t *bmp_data, uint8_t *new_bmp_img, int ht, int wd, int threshold, uint32_t *d_percent_black_cells)
{
	int Gx, Gy;
	uint32_t mag;
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i==0 || j==0 || i == ht -1 || j == wd-1)
	    new_bmp_img[ i*wd + j] = 0;
	
	if(i==0 || j==0||i>=ht-1 || j>=wd-1) {
	    	return;
	}
	   
	Gx = bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i-1)*wd + (j-1) ] \
	+ 2*bmp_data[ (i)*wd + (j+1) ] - 2*bmp_data[ (i)*wd + (j-1) ] \
	+ bmp_data[ (i+1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ];
	
	Gy = bmp_data[ (i-1)*wd + (j-1) ] + 2*bmp_data[ (i-1)*wd + (j) ] \
	+ bmp_data[ (i-1)*wd + (j+1) ] - bmp_data[ (i+1)*wd + (j-1) ] \
	- 2*bmp_data[ (i+1)*wd + (j) ] - bmp_data[ (i+1)*wd + (j+1) ];
	
	mag = sqrtf(Gx * Gx + Gy * Gy);
	
	if(mag > threshold)
	{
		new_bmp_img[ i*wd + j] = 255;
	}
	else
	{
		new_bmp_img[ i*wd + j] = 0;
		atomicAdd(d_percent_black_cells, 1);
	}
}

void cudaSobelTransform()
{
	uint8_t *new_bmp_img_cuda, *d_bmp_data, *d_new_bmp_img;
	uint32_t threshold;
	uint32_t percent_black_cells = 0;
	uint32_t *d_percent_black_cells = 0;
	size_t memSize;
	uint32_t num_th_per_blk, gridX, gridY;
	uint32_t total_cells;
    
	//Allocate new output buffer of same size
    new_bmp_img_cuda = (uint8_t *)malloc(number_of_pixels);
	
	threshold = 0;
	total_cells = wd * ht;

    memSize = number_of_pixels * sizeof(uint8_t);
	
	num_th_per_blk= 32;
	gridX = ht/num_th_per_blk +1;
	gridY = wd/num_th_per_blk + 1;
	
	dim3 dimGrid(gridX, gridY);
	dim3 dimBlock(num_th_per_blk, num_th_per_blk);

	cudaMalloc( (void**) &d_bmp_data, memSize);
	cudaMalloc( (void**) &d_new_bmp_img, memSize);
	cudaMalloc( (void**) &d_percent_black_cells, sizeof(uint32_t));
	cudaMemcpy( d_bmp_data, bmp_data, memSize, cudaMemcpyHostToDevice );

    //===============================Time Calculations=======================================
	struct timespec start_c, end_c; 
	double diff; 
	clock_gettime(CLOCK_REALTIME,& start_c);
	
	clock_t t;
	t = clock();
	//=======================================================================================
	
	while(percent_black_cells < 75)
	{
		percent_black_cells = 0;
		threshold += 1;
		
		cudaMemcpy( d_percent_black_cells, &percent_black_cells, sizeof(uint32_t) , cudaMemcpyHostToDevice );

		sobelEdgeDetect<<< dimGrid, dimBlock>>>(d_bmp_data,d_new_bmp_img,ht,wd,threshold, d_percent_black_cells);
		
		cudaMemcpy( &percent_black_cells, d_percent_black_cells, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		percent_black_cells = (percent_black_cells * 100) / total_cells;
	}
	
	//===============================Time Calculations=======================================
	clock_gettime(CLOCK_REALTIME,& end_c); 
	diff = ((double)end_c.tv_sec + 1.0e-9*end_c.tv_nsec) - ((double)start_c.tv_sec + 1.0e-9*start_c.tv_nsec);
    t = clock() - t;
	//========================================================================================
	
	cudaMemcpy( new_bmp_img_cuda, d_new_bmp_img, memSize,cudaMemcpyDeviceToHost);
	
	printf("%s","***********************************************************************\n");
	printf("CUDA Execution \n");
	printf("Elapsed time for Sobel Operation time : %lf \n",diff);
	printf("Threshold: %d\n",threshold);
	printf("***********************************************************************\n");
	
	//Write the buffer into the bmp file
	write_bmp_file(out_file_Parallel, new_bmp_img_cuda);
	
	cudaFree(d_bmp_data);
    cudaFree(d_new_bmp_img);
	cudaFree(d_percent_black_cells);
	
	free(new_bmp_img_cuda);
}

void serialSobelTransform()
{
    
	uint8_t *new_bmp_img;
	uint32_t i, j;
	uint32_t sum1, sum2, mag;
	uint32_t threshold;
	uint32_t percent_black_cells = 0;
	uint32_t total_cells;
	
	//Allocate space for new sobel image
	new_bmp_img = (uint8_t *)malloc(number_of_pixels);
	threshold   = 0;
	total_cells = wd * ht;
	//===============================Time Calculations=======================================
	struct timespec start_c, end_c; 
	double diff; 
	clock_gettime(CLOCK_REALTIME,& start_c);
	
	clock_t t;
	t = clock();
	//========================================================================================
	
	threshold   = 0;
	total_cells = wd * ht;

	while(percent_black_cells < 75)
	{	
		percent_black_cells = 0;
		threshold += 1;
		
		for(i=1; i < (ht-1); i++)
		{
			for(j=1; j < (wd-1); j++)
			{
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
					new_bmp_img[ i*wd + j] = 0;
					percent_black_cells++;
				}
			}
		}
		percent_black_cells = (1.0 * percent_black_cells) / total_cells * 100;
	}
	
    //===============================Time Calculations========================================
    clock_gettime(CLOCK_REALTIME,& end_c); 
    diff = ((double)end_c.tv_sec + 1.0e-9*end_c.tv_nsec) - ((double)start_c.tv_sec + 1.0e-9*start_c.tv_nsec);
    t = clock() - t;
	//========================================================================================
	
    printf("%s","***********************************************************************\n");
	printf("Serial Execution \n");
	printf("Elapsed time for Sobel Operation time : %lf \n",diff);
	printf("Threshold: %d\n",threshold);
	printf("***********************************************************************\n");
	
	//Write the buffer into the bmp file
	write_bmp_file(out_file_Serial, new_bmp_img);
	//=================================================================
	
}

int main(int argc, char* argv[])
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
                out_file_Serial = fopen(argv[cmd_arg], "wb");
				break;
			case 3:
				out_file_Parallel = fopen(argv[cmd_arg], "wb");;
		}
	}
	
	//Read the binary bmp file into buffer
	bmp_data = (uint8_t *)read_bmp_file(inFile);
	
	//Get image attributes
    wd = get_image_width();
    ht = get_image_height();
	number_of_pixels = get_num_pixel();
	
	printf("%s","***********************************************************************\n");
	printf("SOBEL EDGE DETECTION \n");
	printf("***********************************************************************\n");
	
    serialSobelTransform();
    cudaSobelTransform();

    return 0;
}