#include <stdio.h>
#include <stdlib.h>


__global__ void GPU_Matrix_Multiply_Kernel(double *A, double *B, double *C, int N) {
    	
	//2D Thread ID
    	int tx = threadIdx.x;
    	int ty = threadIdx.y;

    	//Cvalue stores the C element that is computed by the thread
    	float Cvalue = 0;

	//Looping through to compute entries of C
    	for(int k = 0; k < N ; ++k) {

        	float Mdelement = A[ty*N + k];
        	float Ndelement = B[k*N + tx];
        	Cvalue += (Mdelement*Ndelement);
    	}	

	//Storing the value into C
    	C[ty*N + tx] = Cvalue;
}

