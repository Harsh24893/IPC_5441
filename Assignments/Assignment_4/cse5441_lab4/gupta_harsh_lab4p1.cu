#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


//================ Perform matrix transpose multiply on GPU ===========================
__global__ void MatrixTransposeMultiplyDevice(double *d_A, double *CDevice, int matrixSize) {
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k;
	double sum = 0.0;
	for(k = 0; k < matrixSize; k++) {
		sum += d_A[ k * matrixSize + i ] * d_A [ k * matrixSize + j];
	}
	CDevice [ i * matrixSize + j ] = sum;
}
//======================================================================================

//================ Perform matrix transpose multiply on CPU ============================
void MatrixMultiplyHost(double *A, double *CHost, int dim) {
	for (int i = 0; i < dim; i++) 
		for (int j = 0; j < dim; j++) {
		    double sum = 0.0;
			for (int k = 0; k < dim; k++) {
				sum += A[ (k * dim) + i ] * A[ (k * dim) + j ];
			}
			CHost[ i*dim + j] = sum;
		}
}
//======================================================================================

//================== Validate if two matrices are the same =============================
int MatrixTransposeHostValidate(double *A, double *C, int dim)
{
	for (int i = 0; i < dim; i++) 
		for (int j = 0; j < dim; j++) 
			if(C[i * dim + j] != A[i * dim + j]) {
				return 0;
			}
	return 1;
}
//======================================================================================

//================================== Print the matrix ==================================
void printMatrix(double *A, int dim) {
    for(int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%f ",A[i * dim + j]);
        }
        printf("\n");
    }
}
//======================================================================================

//============================= Initialize the matrix ==================================
void initMatrix(double *A, int dim) { 
	for (int i= 0; i< dim; i++){
		for (int j = 0; j < dim; j++) {
			A[i* dim + j] =  (float)(rand()/(float)RAND_MAX) + 1;
		}
	}
}
//======================================================================================

int main(void) {

//============================= Host Code ===================================
	double *A, *CHost, *CD;
	double GFLOPS_host; 
	
	int dim = 1024;
	const double totalOperations = (1.0*dim*dim)*(dim*2);
	size_t memSize = sizeof(double) * dim * dim;
	
	// Allocate memory for the matrices
	A = (double *) malloc(memSize);
	CHost = (double *) malloc(memSize);
	CD = (double *) malloc(memSize);

	// Initialize A
	initMatrix(A, dim);
	
	// Measure the time for the computation on CPU
	double start_time, end_time;
	
	printf("**************************************************\n");
	printf("Matrix multiply on Host(CPU)\n");
	
	// Start time for matrix multiply on CPU
	start_time = rtclock();

	// Perform the matrix transpose on the CPU
	MatrixMultiplyHost(A, CHost, dim);
	
	// End time for matrix transpose on CPU
	end_time = rtclock();

	// Print stats for the CPU
	double time_diff = end_time - start_time;
	printf("Time taken for matrix multiplication on CPU (sec) = %.5f\n", time_diff);
	GFLOPS_host = totalOperations / (1.0e9*time_diff);
	printf("GFLOPS/sec in Host = %f\n", GFLOPS_host);
	//printf("Host Multiply\n");
	//printMatrix(CHost, dim);	
	printf("**************************************************\n");
	printf("\n");
//=============================================================================

	
//============================= Device Code ===================================
	double GFLOPS_device;
	
	double *d_A, *CDevice;
	
	// Define thread hierarchy
	int tpb = 32;
	int nblocks= dim/tpb;
	
	// Start time for matrix transpose on GPU
	start_time = rtclock();
	
	// Allocate device memory
	cudaMalloc( (void**) &d_A, memSize);
	cudaMalloc( (void**) &CDevice, memSize);
	
	// Make a copy of the matrix A to d_A
	cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
    	
	// Launch kernel
	dim3 dimGrid(nblocks, nblocks);
	dim3 dimBlock(tpb, tpb);

	// Perform the matrix transpose on the GPU device	
	MatrixTransposeMultiplyDevice<<< dimGrid, dimBlock>>>(d_A, CDevice, dim);
	
	cudaMemcpy(CD, CDevice, memSize, cudaMemcpyDeviceToHost);

	// End time for matrix transpose on GPU
	end_time = rtclock();
	
	// Print stats for the GPU
	printf("**************************************************\n");
	printf("Matrix multiply on Device(GPU)\n");
	double time_diff_device = end_time - start_time;
	printf("Time taken for matrix multiplication transpose in GPU with %-2d block(s) of %-4d threads (sec)  = %.5f\n", nblocks, tpb, time_diff_device);
	GFLOPS_device = totalOperations / (1.0e9*time_diff_device);
	printf("GFLOPS/sec in Device = %f\n", GFLOPS_device);
	printf("**************************************************\n");
	printf("\n");
	
//=========================================================================

	// Verfiy results between the CPU and GPU
	if(!MatrixTransposeHostValidate(CD, CHost, dim))
		fprintf(stderr, "Wrong results for matrix multiplication on GPU\n");

	// Free memory
	cudaFree(d_A);
	cudaFree(CDevice);
	
	free(A);
	free(CHost);
	free(CD);
}
