#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

void readFromFile(double ***matrix, int *row, int *col, char* file) {
    FILE *fp;
    if ((fp = fopen(file, "r")) == NULL) {
        perror("fopen");
        printf("%s\n", file);
        exit(1);
    }
    // Read the matrix dimensions
    fscanf(fp, "%d\t%d\n", row, col);

    // Allocate memory for the matrix
    *matrix = (double **)malloc((*row) * sizeof(double *));
    for (int i = 0; i < *row; ++i) {
        (*matrix)[i] = (double *)malloc((*col) * sizeof(double));
    }

    // Read the matrix data
    for (int i = 0; i < *row; ++i) {
        for (int j = 0; j < *col; ++j) {
            fscanf(fp, "%lf", &(*matrix)[i][j]);
        }
    }

    fclose(fp);
}

void writeToFile(double **matrix, int rows, int columns, char* file) {
    FILE *fp;
    if ((fp = fopen(file, "w")) == NULL) {
        perror("fopen");
        exit(1);
    }
    fprintf(fp, "%d\t%d\n", rows, columns);  // Write rows and columns as the first line
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            fprintf(fp, "%-10f\t", matrix[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

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

void Orthogonal(double **matrix, int row, int col, int c1, int c2, double **V, bool *pass) {
    double *Ci = (double *)malloc(row * sizeof(double));
    double *Cj = (double *)malloc(row * sizeof(double));

    for (int i = 0; i < row; i++) {
        Ci[i] = matrix[i][c1];
        Cj[i] = matrix[i][c2];
    }
    double inner_prod = vec_mult(Ci, Cj, row);
    if (fabs(inner_prod) < 1e-6) {
        free(Ci);
        free(Cj);
        return;
    }
    *pass = false;

    double len1 = vec_mult(Ci, Ci, row);
    double len2 = vec_mult(Cj, Cj, row);

    if (len1 < len2) {
        for (int i = 0; i < row; ++i) {
            double temp = Ci[i];
            Ci[i] = Cj[i];
            Cj[i] = temp;
        }
    }

    double tao = (len1 - len2) / (2 * inner_prod);
    double tan = sign(tao) / (fabs(tao) + sqrt(1 + tao * tao));
    double cos = 1 / sqrt(1 + tan * tan);
    double sin = cos * tan;

    for (int i = 0; i < row; i++) {
        double var1 = Ci[i] * cos + Cj[i] * sin;
        double var2 = Cj[i] * cos - Ci[i] * sin;
        Ci[i] = var1;
        Cj[i] = var2;
    }

    for (int i = 0; i < row; i++) {
        matrix[i][c1] = Ci[i];
        matrix[i][c2] = Cj[i];
    }

    for (int i = 0; i < col; i++) {
        double var1 = V[i][c1] * cos + V[i][c2] * sin;
        double var2 = V[i][c2] * cos - V[i][c1] * sin;
        V[i][c1] = var1;
        V[i][c2] = var2;
    }

    free(Ci);
    free(Cj);
}

void jacob_one_side(double **matrix, int row, int col, double **V) {
    int iterat = 100;
    while (iterat-- > 0) {
        bool pass = true;
        for (int i = 0; i < col; ++i) {
            for (int j = i + 1; j < col; ++j) {
                Orthogonal(matrix, row, col, i, j, V, &pass);
            }
        }
        if (pass) break;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_matrix_file>\n", argv[0]);
        return 1;
    }

    int ROW, COL;
    double **A;

    // Read matrix from file
    readFromFile(&A, &ROW, &COL, argv[1]);

    // Allocate memory for V
    double **V = (double **)malloc(COL * sizeof(double *));
    for (int i = 0; i < COL; ++i) {
        V[i] = (double *)malloc(COL * sizeof(double));
        for (int j = 0; j < COL; ++j) {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    jacob_one_side(A, ROW, COL, V);

    // Calculate singular values (S matrix)
    double **S = (double **)malloc(ROW * sizeof(double *));
    for (int i = 0; i < ROW; ++i) {
        S[i] = (double *)calloc(COL, sizeof(double));
    }

    double *E = (double *)calloc(COL, sizeof(double));
    for (int i = 0; i < COL; ++i) {
        double norm = 0;
        for (int j = 0; j < ROW; ++j) {
            norm += A[j][i] * A[j][i];
        }
        E[i] = sqrt(norm);
        if (i < ROW) {
            S[i][i] = E[i];
        }
    }

    // Calculate U matrix
    double **U = (double **)malloc(ROW * sizeof(double *));
    for (int i = 0; i < ROW; ++i) {
        U[i] = (double *)calloc(ROW, sizeof(double));
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            if (E[j] > 1e-6) {
                U[i][j] = A[i][j] / E[j];
            }
        }
    }

    // Write results to file
    writeToFile(A, ROW, COL, "A_output.txt");
    writeToFile(S, ROW, COL, "S_output.txt");
    writeToFile(V, COL, COL, "V_output.txt");
    writeToFile(U, ROW, ROW, "U_output.txt");

    // Free memory
    for (int i = 0; i < ROW; ++i) {
        free(A[i]);
        free(S[i]);
        free(U[i]);
    }
    for (int i = 0; i < COL; ++i) {
        free(V[i]);
    }
    free(A);
    free(S);
    free(U);
    free(V);
    free(E);

    return 0;
}