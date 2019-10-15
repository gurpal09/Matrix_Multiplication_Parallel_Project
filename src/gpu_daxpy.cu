#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>


void GPU_daxpy(int m, int n, int k, double *acol, double *bcol, double *c){
	
	//Initializing alpha variable for daxpy	
	double alpha;

	//Space to store rows of A and columns of B to pass
	double *a_p;
	cudaMallocManaged( &a_p,m * sizeof(double));
	
	double *b_p;
	cudaMallocManaged( &b_p,n * sizeof(double));
	
	//Initialize to zero
	for(int i =0; i<n; i++){
		b_p[i] = 0;
	}

	//Space to store result
	double *result;
	cudaMallocManaged( &result,m * sizeof(double));	
	
	//Looping through values of alpha and columns of A
	for(int z = 0; z < k; z++){
		for(int u = 0; u < m; u++){
			
			result[u] = 0;
		}	 	
		
		for(int i = 0; i < n; i++){	
			
			b_p[i] = bcol[z*n+i];

			for(int j = 0; j < m; j++){
				
				a_p[j] = acol[i*m+j];	
			}
		
			alpha = b_p[i];
			cublasDaxpy(m, alpha, a_p, 1, result, 1);
			cudaDeviceSynchronize();		
		}
			
		//Store result to C here.
		for(int i = 0; i < m; i++){
			c[z*m+i] = result[i];	
		}
	}

	//Converting back to row major
	double *c_row;
	cudaMallocManaged( &c_row, m * k * sizeof(double));

	for(int i = 0; i < k; i++){
		for(int j = 0; j < m; j++){

			c_row[j*k+i] = c[i*m+j];
		}
	}	

	//Clean Up
	cudaFree(a_p);
	cudaFree(b_p);
	cudaFree(result);
	cudaFree(c_row);
}
