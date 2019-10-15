#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>

void GPU_ddot(int m, int n, int k, double *a, double *bcol, double *c){

	int incx = 1;
	int incy = 1;

	//Loop to compute entries of C using cublas
	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
			
			c[k*i+j] = cublasDdot(n, &a[n*i], incx, &bcol[n*j], incy);
			cudaDeviceSynchronize();
		}
	}	
}

