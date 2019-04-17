/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <stdlib.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int row = 0; row < n; ++row) {
    	double sum = 0;

    	for (int c = 0; c < n; ++c) {
    		sum += A[row * n + c] * x[c];
    	}

    	y[row] = sum;
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int row = 0; row < n; ++row) {
    	double sum = 0;

    	for (int c = 0; c < m; ++c) {
    		sum += A[row * n + c] * x[c];
    	}

    	y[row] = sum;
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
	int n_squared = n * n;

	//There are only n values of import in the diagonal matrix
	double* diagonal = (double *) malloc(n * sizeof(double));

	//R is an nxn matrix
    double* R = (double *) malloc(n_squared * sizeof(double));

    //Initializing R matrix and diagonal matrix by looping through A
    for (int i = 0; i < n; ++i) {

    	for (int j = 0; j < n; ++j) {
    		if (j == i) {
    			//We are on the diagonal of A
    			diagonal[i] = A[i * n + j];
    			R[i * n + j] = 0;
    		} else {
    			//Not on the diagonal of A
    			R[i * n + j] = A[i * n + j];
    		}


    	}

    }
    
    double* result_of_R_times_x = (double *) malloc(n * sizeof(double));

    for (int i = 0; i < max_iter; i++) {

    	//We run the jacobi method
    	//Rx matrix vector multiplication first
    	matrix_vector_mult(n, R, x, result_of_R_times_x);

    	//Iterate throw x vector. This updates our guess
    	for (int row = 0; row < n; ++row) {

    		x[row] = (b[row] - result_of_R_times_x[row]) / diagonal[row];

    	}


    	//We calculate the l2 norm
    	double sum_of_squares = 0;

    	for (int row = 0; row < n; ++row) {
    		sum_of_squares += pow(x[row], 2.0);
    	}

    	double l2_norm = sqrt(sum_of_squares);

    	if (l2_norm < l2_termination) 
        {
            break;
        }
    }


    free(diagonal);
    free(R);
    free(result_of_R_times_x);
}
