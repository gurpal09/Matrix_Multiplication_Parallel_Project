#include <stdio.h>
#include <stdlib.h>

void CPU_Matrix_Multiply(int m, int n, int k, double *a, double *b, double *c){

	for (int x = 0; x < m; x++) { // row number of output
    		
		for (int y = 0; y < k; y++) { // column number of output
        		
			c[k*x+y] = 0;
			
			for (int z = 0; z < n; z++) { //Add n elements
				
				c[k*x+y] += a[n*x+z] * b[k*z+y];
        		}		
    		}	
	}
}
