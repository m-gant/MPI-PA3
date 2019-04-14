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


        int root;
        int root_coordinates[2] = {0,0};
        MPI_Cart_rank(column_comm, root_coordinates, &root);

        MPI_Scatterv(input_vector, send_counts, displacements,
            MPI_DOUBLE, *local_vector, 100, MPI_DOUBLE, root, column_comm);

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

    int p, global_rank;

    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &global_rank);

    int q = sqrt(p);

    int cartesian_coords[2];
    MPI_Cart_coords(comm, global_rank, 2, cartesian_coords);


    int row_size = cartesian_coords[0] < (n % q) ? ceil(((double)n) / q) : floor(((double)n) /q);
    int column_size = cartesian_coords[1] < (n%q) ? ceil(((double)n) / q) : floor(((double)n) / q);

    int matrix_size = row_size * column_size;
    *local_matrix = (double *) malloc(sizeof(double) * matrix_size);



    //We are at (0,0). We will build matrices to send to other processes
    if (cartesian_coords[0] == 0 && cartesian_coords[1] == 0) {



        int row_offset = 0;
        int col_offset = 0;

        for (int row = 0; row < q; ++row) {
            int amt_of_sending_rows = row < (n % q) ? ceil(((double)n) / q) : floor(((double)n) /q);
            
            for (int column = 0; column < q; ++column) {
                int destination_rank;

                int destination_coordinates[2] = {row, column};

                MPI_Cart_rank(comm, destination_coordinates, &destination_rank);

                
                int amt_of_sending_column = column < (n % q) ? ceil(((double)n) / q) : floor(((double)n) /q);

                int size_of_sending_matrix = amt_of_sending_rows * amt_of_sending_column;
                double * matrix_to_send = (double *) malloc(sizeof(double) * size_of_sending_matrix);

                int index_of_sending_matrix = 0;
                //Build matrix to send to processor
                for (int i = 0; i < amt_of_sending_rows; ++i) {

                    for (int j = 0; j < amt_of_sending_rows; ++j) {
                        matrix_to_send[index_of_sending_matrix++] = input_matrix[row_offset * n  + (i * n) + j + col_offset];
                    }

                }


                col_offset += amt_of_sending_column;

                MPI_Send(matrix_to_send, size_of_sending_matrix, MPI_DOUBLE, destination_rank, 200, comm);


                free(matrix_to_send);


            }

            row_offset += amt_of_sending_rows;
            col_offset = 0;
        }

    }

    int root_rank;
    int root_coordinates[2] = {0,0};
    MPI_Cart_rank(comm, root_coordinates, &root_rank);

    MPI_Status status;
    MPI_Recv(*local_matrix, matrix_size, MPI_DOUBLE, root_rank, 200, comm, &status);

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
