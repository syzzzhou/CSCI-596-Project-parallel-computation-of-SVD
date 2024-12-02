#ifndef SVD_JACOB_H
#define SVD_JACOB_H

#include <stdbool.h>
#include <stdio.h>

extern double THRESHOLD;
extern int ITERATION;
extern int ROW;
extern int COL;

void print_matrix(double **A, int r, int c, const char *name);
double vec_mult(double *v1, double *v2, int len);
void Orthogonal(double** matrix, int rows, int c1, int c2, double** V, bool *pass);
void jacob_one_side_mpi(double** matrix, int rows, int columns, double** V, int rank, int size);
double** allocate_matrix(int rows, int columns);
void free_matrix(double** matrix, int rows);

#endif