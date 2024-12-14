#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <cuda.h>

#define ROW 6  // Example row size
#define COL 3  // Example column size
#define ITERATION 100
#define THRESHOLD 1e-6
#define NUM_BLOCK 5  // Number of thread blocks
#define NUM_THREAD ((ROW + NUM_BLOCK - 1) / NUM_BLOCK)  // Threads per block, dividing rows among blocks

// CUDA kernel for orthogonalization
__global__ void orthogonal_kernel(double *matrix, double *V, bool *pass, int row, int col, int c1, int c2) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdx >= row) return;

    __shared__ double Ci[ROW];
    __shared__ double Cj[ROW];

    // Load columns into shared memory
    if (globalIdx < row) {
        Ci[globalIdx] = matrix[globalIdx * col + c1];
        Cj[globalIdx] = matrix[globalIdx * col + c2];
    }
    __syncthreads();

    // Calculate inner product and lengths
    double inner_prod = 0, len1 = 0, len2 = 0;
    for (int i = 0; i < row; ++i) {
        inner_prod += Ci[i] * Cj[i];
        len1 += Ci[i] * Ci[i];
        len2 += Cj[i] * Cj[i];
    }

    if (fabs(inner_prod) < THRESHOLD) {
        if (globalIdx == 0) *pass = true;
        return;
    }
    if (globalIdx == 0) *pass = false;

    if (len1 < len2) {
        for (int i = 0; i < row; ++i) {
            double temp = Ci[i];
            Ci[i] = Cj[i];
            Cj[i] = temp;
        }
    }

    double tao = (len1 - len2) / (2 * inner_prod);
    double tan = (tao > 0 ? 1 : -1) / (fabs(tao) + sqrt(1 + tao * tao));
    double cos = 1 / sqrt(1 + tan * tan);
    double sin = cos * tan;

    for (int i = 0; i < row; ++i) {
        double var1 = Ci[i] * cos + Cj[i] * sin;
        double var2 = Cj[i] * cos - Ci[i] * sin;
        Ci[i] = var1;
        Cj[i] = var2;
    }

    for (int i = 0; i < row; ++i) {
        matrix[i * col + c1] = Ci[i];
        matrix[i * col + c2] = Cj[i];
    }
}

int main(int argc, char **argv) {
    double A[ROW][COL] = {
        {6, 5, 1},
        {9, 8, 4},
        {8, 5, 2},
        {4, 6, 9},
        {1, 2, 3},
        {2, 1, 4}};
    double V[COL][COL] = {0};
    double S[ROW][COL] = {0};
    double U[ROW][ROW] = {0};
    bool pass;

    for (int i = 0; i < COL; ++i) {
        V[i][i] = 1.0;
    }

    double *d_A, *d_V;
    bool *d_pass;
    size_t matrix_size = ROW * COL * sizeof(double);
    size_t vector_size = COL * COL * sizeof(double);

    // Allocate device memory
    cudaMalloc((void **)&d_A, matrix_size);
    cudaMalloc((void **)&d_V, vector_size);
    cudaMalloc((void **)&d_pass, sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, vector_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(NUM_THREAD);
    dim3 dimGrid(NUM_BLOCK);

    for (int iter = 0; iter < ITERATION; ++iter) {
        pass = true;
        cudaMemcpy(d_pass, &pass, sizeof(bool), cudaMemcpyHostToDevice);

        for (int i = 0; i < COL; ++i) {
            for (int j = i + 1; j < COL; ++j) {
                orthogonal_kernel<<<dimGrid, dimBlock>>>(d_A, d_V, d_pass, ROW, COL, i, j);
                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(&pass, d_pass, sizeof(bool), cudaMemcpyDeviceToHost);
        if (pass) break;
    }

    // Copy results back to host
    cudaMemcpy(A, d_A, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, vector_size, cudaMemcpyDeviceToHost);

    // Calculate singular values (S matrix)
    double E[COL] = {0};
    int nonzero = 0;
    for (int i = 0; i < COL; ++i) {
        double norm = 0;
        for (int j = 0; j < ROW; ++j) {
            norm += A[j][i] * A[j][i];
        }
        if (norm > THRESHOLD) nonzero++;
        E[i] = sqrt(norm);
    }

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            S[i][j] = (i == j) ? E[i] : 0.0;
        }
    }

    // Calculate U matrix
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < nonzero; ++j) {
            U[i][j] = A[i][j] / E[j];
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_V);
    cudaFree(d_pass);

    // Print results
    printf("Matrix A (after orthogonalization):\n");
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    printf("Matrix S:\n");
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            printf("%f ", S[i][j]);
        }
        printf("\n");
    }

    printf("Matrix V:\n");
    for (int i = 0; i < COL; ++i) {
        for (int j = 0; j < COL; ++j) {
            printf("%f ", V[i][j]);
        }
        printf("\n");
    }

    printf("Matrix U:\n");
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < ROW; ++j) {
            printf("%f ", U[i][j]);
        }
        printf("\n");
    }

    return 0;
}
