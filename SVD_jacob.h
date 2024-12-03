#ifndef SVD_JACOB_H
#define SVD_JACOB_H

extern double THRESHOLD;
extern int ITERATION;
extern int ROW;
extern int COL;

void print_matrix(double *A, int r, int c);
void jacob_one_side(double matrix[ROW][COL], double V[COL][COL]);

#endif // SVD_JACOB_H