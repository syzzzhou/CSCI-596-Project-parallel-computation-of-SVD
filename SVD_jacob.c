#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include "SVD_jacob.h"

double THRESHOLD = 1E-8;
int ITERATION = 30;
int ROW = 2;
int COL = 3;

int sign(double number) {
    return (number < 0) ? -1 : 1;
}

double vec_mult(double *v1, double *v2, int len) {
    double val = 0;
    for (int i = 0; i < len; i++) {
        val += v1[i] * v2[i];
    }
    return val;
}

void Orthogonal(double** matrix, int rows, int c1, int c2, double** V, bool *pass) {
    double* Ci = (double*)malloc(rows * sizeof(double));
    double* Cj = (double*)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        Ci[i] = matrix[i][c1];
        Cj[i] = matrix[i][c2];
    }
    double inner_prod = vec_mult(Ci, Cj, rows);
    if (fabs(inner_prod) < THRESHOLD) {
        free(Ci);
        free(Cj);
        return;
    }
    *pass = false;

    double len1 = vec_mult(Ci, Ci, rows);
    double len2 = vec_mult(Cj, Cj, rows);

    if (len1 < len2) {
        for (int row = 0; row < rows; ++row) {
            matrix[row][c1] = Cj[row];
            matrix[row][c2] = Ci[row];
        }
        for (int row = 0; row < COL; ++row) {
            double tmp = V[row][c1];
            V[row][c1] = V[row][c2];
            V[row][c2] = tmp;
        }
    }

    double tao = (len1 - len2) / (2 * inner_prod);
    double tan = sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    double cos = 1 / sqrt(1 + pow(tan, 2));
    double sin = cos * tan;

    for (int row = 0; row < rows; ++row) {
        double var1 = matrix[row][c1] * cos + matrix[row][c2] * sin;
        double var2 = matrix[row][c2] * cos - matrix[row][c1] * sin;
        matrix[row][c1] = var1;
        matrix[row][c2] = var2;
    }
    for (int col = 0; col < COL; ++col) {
        double var1 = V[col][c1] * cos + V[col][c2] * sin;
        double var2 = V[col][c2] * cos - V[col][c1] * sin;
        V[col][c1] = var1;
        V[col][c2] = var2;
    }

    free(Ci);
    free(Cj);
}

void jacob_one_side_mpi(double** matrix, int rows, int columns, double** V, int rank, int size) {
    int iterat = ITERATION;
    bool global_pass;

    while (iterat-- > 0) {
        bool local_pass = true;

        if (rank == 0) {
            printf("Rank %d: Starting iteration %d\n", rank, ITERATION - iterat);
        }

        // 每次迭代时，对所有列对进行正交化
        for (int i = 0; i < columns; ++i) {
            for (int j = i + 1; j < columns; ++j) {
                if ((i + j) % size == rank) {  // 每个进程负责特定的列对
                    printf("Rank %d: Processing columns %d and %d\n", rank, i, j);
                    Orthogonal(matrix, rows, i, j, V, &local_pass);
                }
                
                // 确保每对列的正交化完成后，同步所有进程
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        // 合并所有进程的 pass 标志，判断是否所有列对都已经正交
        MPI_Allreduce(&local_pass, &global_pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Rank %d: Global pass after Allreduce: %d\n", rank, global_pass);
        }

        // 如果所有列对都已经正交化，则退出迭代
        if (global_pass) {
            break;
        }

        // 确保所有进程在这一轮迭代结束后同步
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Number of iterations: %d\n", ITERATION - iterat);
    }
}

void print_matrix(double **A, int r, int c, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

double** allocate_matrix(int rows, int columns) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; ++i) {
        matrix[i] = (double*)calloc(columns, sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}