/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream> 

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{

    //We figure out p and q
    int p, global_rank;

    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &global_rank);

    int q = sqrt(p);
    
    //Find the cartesian coordinates from the global rank
    int cartesian_coords[2];
    MPI_Cart_coords(comm, global_rank, 2, cartesian_coords);

    //We split communicator along the columns
    int color = cartesian_coords[1];
    MPI_Comm column_comm;

    MPI_Comm_split(comm, color, cartesian_coords[0], &column_comm);

    int column_rank, column_size;
    MPI_Comm_rank(column_comm, &column_rank);
    MPI_Comm_size(column_comm, &column_size);


    //We send out the vector along the first column
    if (cartesian_coords[1] == 0) {
        
        //Setting up a scatterv call. Sendcounts describes how many elements will be
        //sent to each processor and displacements is an array describing the beginning of each segment
        int* send_counts = (int *) malloc(sizeof(int) * q);
        int* displacements = (int *) malloc(sizeof(int) * q);

        int current_displacement = 0; //used to calculate displacement

        //iterate through each row in grid column 0. Calculate amount of vector elements
        //to send based off specs
        for (int r = 0; r < q; ++r) {

            //Calculates the local vector size based off row
            int local_vector_size = r < (n % q) ? ceil( ((double)n) / q) : floor( ((double)n) / q);

            send_counts[r] = local_vector_size;
            displacements[r] = current_displacement;
            current_displacement += local_vector_size;
        }



        int local_vector_size = cartesian_coords[0] < (n %q) ? ceil( ((double)n) / q) : floor(((double)n) /q);
        *local_vector = (double *) malloc(sizeof(double) * local_vector_size);

        MPI_Scatterv(input_vector, send_counts, displacements,
            MPI_DOUBLE, *local_vector, 100, MPI_DOUBLE, 0, column_comm);

        free(send_counts);
        free(displacements);
    }


}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // TODO
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{



    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    // double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    // distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // // gather results back to rank 0
    // gather_vector(n, local_x, x, comm);
}
