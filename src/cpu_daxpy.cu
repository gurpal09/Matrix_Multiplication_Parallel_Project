#include <stdio.h>
#include <stdlib.h>
#include "gsl_cblas.h"



void CPU_daxpy(int m, int n, int k, double *acol, double *bcol, double *c){

	//Initializing variable for daxpy	
	double alpha;

	//Space to store the row of A and the column of B that will be passed
	double *a_p;
	cudaMallocManaged( &a_p,m * sizeof(double));
	
	double *b_p;
	cudaMallocManaged( &b_p,n * sizeof(double));

	//For-loop to initialize b_p to zero
	for(int i =0; i<n; i++){
		b_p[i] = 0;
	}
	
	for(int i = 0; i<k; i++){
		for(int j = 0; j<n; j++){
		
			b_p[j] = bcol[i*n+j];
		}
	}

	//Space for the result
	double *result;
	cudaMallocManaged( &result,m * sizeof(double));

	//Initialize it to zero
	for(int i = 0; i < m; i++){
		result[i] = 0;
	} 	
	
	
	//Looping through the value of alpha and the columns of A and summing
	for(int z = 0; z < k; z++){//k
	
		for(int u = 0; u < m; u++){
			result[u] = 0;
		}	 	
	
			for(int i = 0; i < n; i++){	
				b_p[i] = bcol[z*n+i];
		
					for(int j = 0; j < m; j++){
						a_p[j] = acol[i*m+j];	
					}

				alpha = b_p[i];

				//daxpy function call
				cblas_daxpy(m, alpha, a_p, 1, result, 1);				    }
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
