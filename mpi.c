#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "SVD_jacob.h"

#define THRESHOLD 1e-8
#define ITERATION 20
#define ROW 3
#define COL 3

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Rank %d: Starting main function with %d processes\n", rank, size);
    }

    if (ROW > COL) {
        MPI_Finalize();
        return 0;
    }

    // 分配矩阵 A, V, S, U
    double** A = allocate_matrix(ROW, COL);
    double vec[] = {3, 2, 2, 2, 3, -2, 1, 1, 1};
    double** V = allocate_matrix(COL, COL);
    double** S = allocate_matrix(ROW, COL);
    double** U = allocate_matrix(ROW, ROW);

    int index = 0;
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            A[i][j] = vec[index++];
        }
    }

    print_matrix(A, ROW, COL, "Initial Matrix A");

    for (int i = 0; i < COL; i++) {
        V[i][i] = 1.0;
    }

    jacob_one_side_mpi(A, ROW, COL, V, rank, size);

    double E[COL];
    int nonzero = 0;
    for (int i = 0; i < COL; ++i) {
        double norm = 0;
        for (int j = 0; j < ROW; j++) {
            norm += A[j][i] * A[j][i];
        }
        if (norm > THRESHOLD)
            nonzero += 1;
        E[i] = sqrt(norm);
    }

    for (int i = 0; i < ROW; ++i) {
        S[i][i] = E[i];
        for (int j = 0; j < nonzero; j++) {
            U[i][j] = A[i][j] / E[j];
        }
    }

    if (rank == 0) {
        print_matrix(A, ROW, COL, "A after Jacobi rotation");
        print_matrix(S, ROW, COL, "Sigma");
        print_matrix(V, COL, COL, "V");
        print_matrix(U, ROW, ROW, "U");
    }

    free_matrix(A, ROW);
    free_matrix(U, ROW);
    free_matrix(V, COL);
    free_matrix(S, ROW);

    MPI_Finalize();
    return 0;
}