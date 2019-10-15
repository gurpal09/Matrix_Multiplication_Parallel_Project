#include <stdio.h>
#include <stdlib.h>
#include "gsl_cblas.h"

void CPU_ddot(int m, int n, int k, double *a, double *bcol, double *c){

	//Initializing incrementation for cblas arguements
	int incx = 1;
	int incy = 1;

	//Loop to compute entries of C using cblas
	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
			
			c[k*i+j] = cblas_ddot(n, &a[n*i], incx, &bcol[n*j], incy);	            }
	}
}

