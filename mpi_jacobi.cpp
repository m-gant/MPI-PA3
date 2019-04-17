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


        MPI_Scatterv(input_vector, send_counts, displacements,
            MPI_DOUBLE, *local_vector, 100, MPI_DOUBLE, 0, column_comm);
    }


}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
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

    //We recv all vectors along the first column
    if (cartesian_coords[1] == 0) {


        //Setting up a gatherv call. Recvcounts describes how many elements are on each proc
        //and displacements is an array describing the beginning of each segment in the recvbuff
        int* recv_counts = (int *) malloc(sizeof(int) * q);
        int* displacements = (int *) malloc(sizeof(int) * q);

        int current_displacement = 0; //used to calculate displacement

        //iterate through each row in grid column 0. Calculate amount of vector elements
        //to send based off specs
        for (int r = 0; r < q; ++r) {

            //Calculates the local vector size based off row
            int other_local_vector_size = r < (n % q) ? ceil( ((double)n) / q) : floor( ((double)n) / q);

            recv_counts[r] = other_local_vector_size;
            displacements[r] = current_displacement;
            current_displacement += other_local_vector_size;
        }

        // std::cout<< "local vector displacements created" << std::endl;

        int this_local_vector_size = cartesian_coords[0] < (n % q) ? ceil( ((double)n) / q) : floor( ((double)n) / q);

        MPI_Gatherv(local_vector, this_local_vector_size, MPI_DOUBLE, output_vector,
            recv_counts, displacements, MPI_DOUBLE, 0, column_comm);

        free(recv_counts);
        free(displacements);
    }
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

    double * matrix_to_send = NULL;


    //We are at (0,0). We will build matrices to send to other processes
    if (cartesian_coords[0] == 0 && cartesian_coords[1] == 0) {


        int row_offset = 0;
        int col_offset = 0;

        for (int row = 0; row < q; row++) {
            int amt_of_sending_rows = row < (n % q) ? ceil(((double)n) / q) : floor(((double)n) /q);

            for (int column = 0; column < q; column++) {

                int destination_rank;

                int destination_coordinates[2] = {row, column};

                MPI_Cart_rank(comm, destination_coordinates, &destination_rank);


                int amt_of_sending_column = column < (n % q) ? ceil(((double)n) / q) : floor(((double)n) /q);

                int size_of_sending_matrix = amt_of_sending_rows * amt_of_sending_column;
                matrix_to_send = (double *) malloc(sizeof(double) * size_of_sending_matrix);

                int index_of_sending_matrix = 0;

                //Build matrix to send to processor
                for (int i = 0; i < amt_of_sending_rows; ++i) {

                    for (int j = 0; j < amt_of_sending_rows; ++j) {
                        matrix_to_send[index_of_sending_matrix++] = input_matrix[row_offset * n  + (i * n) + j + col_offset];
                    }

                }

                col_offset += amt_of_sending_column;

                if (row == 0 && column == 0) {
                    *local_matrix = matrix_to_send;
                } else {
                    MPI_Send(matrix_to_send, size_of_sending_matrix, MPI_DOUBLE, destination_rank, 200, comm);
                }
            }

            row_offset += amt_of_sending_rows;
            col_offset = 0;
        }


    } else {

        int root_rank;
        int root_coordinates[2] = {0,0};
        MPI_Cart_rank(comm, root_coordinates, &root_rank);

        MPI_Status status;
        MPI_Recv(*local_matrix, matrix_size, MPI_DOUBLE, root_rank, 200, comm, &status);
    }

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    //get the global rank and rank in cartesian
    int p, global_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &global_rank);


    //std::cout << "•••••••••••••• number of processors = " << p << std::endl;
    int q = sqrt(p);

    int cartesian_coords[2];
    MPI_Cart_coords(comm, global_rank, 2, cartesian_coords);

    int row_i = cartesian_coords[0];
    int col_j = cartesian_coords[1];

    //create 2 separate communicators for the rows and for the columns
    MPI_Comm column_comm, row_comm;
    MPI_Comm_split(comm, col_j, row_i, &column_comm);
    MPI_Comm_split(comm, row_i, col_j, &row_comm);



    //if we are in column 1 we send our col_vector to the processor (i,i) in our rows
    int num_rows = row_i < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);
    int num_cols = col_j < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);

    if (col_j == 0) {
      // ` << "%%%%%%%%%%%%%%%%%% col_vector = "<< std::endl;
      // for (int i = 0; i < num_rows; i++) {
      //   std::cout << col_vector[i] << std::endl;
      // }
      MPI_Send(col_vector, num_rows, MPI_DOUBLE, row_i, 300, row_comm);
    }


    //if diagonal processor, I receive the elements from first column and broadcast the value along the column comm.
    if (row_i == col_j) {
      MPI_Status status;

      MPI_Recv(row_vector, num_cols, MPI_DOUBLE, 0, 300, row_comm, &status);
      // std::cout << " ^^^^^^ this is what whas received ^^^^^ -> " << std::endl;
      // for (int i = 0; i < num_cols; i++) {
      //   std::cout << row_vector[i] << std::endl;
      // }

    }

    MPI_Bcast(row_vector, num_cols, MPI_DOUBLE, col_j, column_comm);

    // std::cout << " ****** this is what whas received ****** -> " << std::endl;
    // for (int i = 0; i < num_cols; i++) {
    //   std::cout << row_vector[i] << std::endl;
    // }

}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{

    int p, global_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &global_rank);

    int q = sqrt(p);

    int cartesian_coords[2];
    MPI_Cart_coords(comm, global_rank, 2, cartesian_coords);

    int row_i = cartesian_coords[0];
    int col_j = cartesian_coords[1];

    MPI_Comm row_comm;
    MPI_Comm_split(comm, row_i, col_j, &row_comm);

    //transpose local x onto all processors
    int num_cols = col_j < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);
    int num_rows = row_i < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);
    double* row_decomposed_local_x = (double*) malloc(sizeof(double) * num_cols);
    double* unreduced_local_y = (double*) malloc(sizeof(double) * num_rows);
    transpose_bcast_vector(n, local_x, row_decomposed_local_x, comm);

    //multiply row-decomposed vector by the local matrices
    matrix_vector_mult(num_rows, num_cols, local_A, row_decomposed_local_x, unreduced_local_y);
    free(row_decomposed_local_x);

    //reduce along rows

    MPI_Reduce(unreduced_local_y, local_y, num_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    free(unreduced_local_y);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int p, global_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &global_rank);

    int q = sqrt(p);

    int cartesian_coords[2];
    MPI_Cart_coords(comm, global_rank, 2, cartesian_coords);

    int row_i = cartesian_coords[0];
    int col_j = cartesian_coords[1];

    MPI_Comm row_comm, column_comm;
    MPI_Comm_split(comm, row_i, col_j, &row_comm);
    MPI_Comm_split(comm, col_j, row_i, &column_comm);

    int num_cols = col_j < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);
    int num_rows = row_i < (n % q) ? ceil(((double)n) / q) : floor(((double)n) / q);

    double* local_R = (double*) malloc(sizeof(double) * num_cols * num_rows);
    
    for (int i = 0; i < (num_rows * num_cols); i++) {
        //R = deepcopy(A)


        local_R[i] = local_A[i];
    }

    double* local_D;

    //block distribute D to first column & calculate R: set diagonal to zero
    if (row_i == col_j ) {
      local_D = (double*) malloc(sizeof(double) * num_rows);
      for (int i = 0; i < num_rows; i++) {
        local_D[i] = local_A[(i * num_rows) + i]; //diagonal processors have square matrices
        local_R[(i * num_rows) + i] = 0;
      }
      //send diagonal elements to column 0
      MPI_Send(local_D, num_rows, MPI_DOUBLE, 0, 400, row_comm);

    }

    //if in column zero, receive the diagonal elements and initialize x to 0
    if (col_j == 0) {
      local_D = (double*) malloc(sizeof(double) * num_rows);
      MPI_Status status;
      MPI_Recv(local_D, num_rows, MPI_DOUBLE, row_i, 400, row_comm, &status);

      //initialize x to [0 0 ... 0] block distributed along first columns
      for (int i = 0; i < num_rows; i++) {
        local_x[i] = 0;
      }
    }

    // start 'while' loop
    for (int iter = 0; iter < max_iter; iter++) {

      double* local_p = (double*) malloc(sizeof(double) * num_rows);
      double* local_w = (double*) malloc(sizeof(double) * num_rows);

      //p = R*x
      distributed_matrix_vector_mult(n, local_R, local_x, local_p, comm);

      //x = (b-p) / D
      if (col_j == 0) {
        for (int i = 0; i < num_cols; i++) {
          local_x[i] = (local_b[i] - local_p[i]) / local_D[i];
        }
      }

      //w = A*x
      distributed_matrix_vector_mult(n, local_A, local_x, local_w, comm);

      //if in column 0, p = (b-w)^2, otherwise p = 0 (helps with Allreduce)
      double l2_squared = 0;

      if (col_j == 0) {
        for (int i = 0; i < num_rows; i++) {
          local_p[i] = (local_b[i] - local_w[i]) * (local_b[i] - local_w[i]);
          l2_squared += local_p[i];
        }
      }

      //l2^2 = SUM(p's in column 0)
      double global_l2;
      MPI_Allreduce(&l2_squared, &global_l2, 1, MPI_DOUBLE, MPI_SUM, comm);




      free(local_p);
      free(local_w);


      if (sqrt(global_l2) <= l2_termination) {
        free(local_D);
        free(local_R);
        return;
      }


    }

    free(local_D);
    free(local_R);


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
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
