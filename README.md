# What is SVD (Singular Value Decomposition)

Given a matrix  M  of size  m x n , the SVD decomposes M into the product of three matrices:


M = U Σ V^T



Where:
- U: An m x m  orthogonal matrix (columns are orthonormal eigenvectors of M M^T 
- Σ: An  m x n  diagonal matrix containing the singular values (non-negative values sorted in decreasing order)
- V : An n x n  orthogonal matrix rows are orthonormal eigenvectors of  M^T M
<img width="240" alt="image" src="https://github.com/user-attachments/assets/8246c7fa-cbf7-47e4-a656-976e1252ade3">


---



## Why It Is Important

SVD is essential for:
- Dimensionality reduction
- Noise filtering
- Matrix approximations

It is widely applied in data science and machine learning. Parallel SVD is crucial for handling large-scale datasets efficiently, enabling faster computation and scalability in high-dimensional applications.

---

## Method

Our project focuses on implementing a parallelized Singular Value Decomposition (SVD) using the **one-sided Jacobi method**.

### Core of Jacobi Method

The method transforms the symmetric matrix M into a diagonal matrix M', whose diagonal entries are the eigenvalues of M M^T , using a sequence of rotations.

The rotation matrix G(i, j, θ) is defined as:

<img width="305" alt="image" src="https://github.com/user-attachments/assets/2299c185-df3e-4ee6-9061-02742148cc20">


This rotational matrix operates only on the i-th and j -th rows/columns to make them orthogonal to each other. Sequentially, we can make all the columns orthogonal to each other. In this way, we would be able to retrieve U* Σ , and our final rotational matrix would be V.

---

### Parallelization Strategy

The independence of Jacobi rotations makes the method highly amenable to parallel computation. Each rotation involves only two columns, and different column pairs can be processed simultaneously.

### Tools:
- **OpenMP**
- **MPI**
- **CUDA**
- **Hybrid Parallelization**

By benchmarking each parallelization approach and their hybrid implementation, the project provides insights into the efficiency of these methods for large-scale SVD computations.

### The division of work
In our work, Yuxiao is mainly in charge of finding the algorithm, implementing the code for one thread, and openmp implementation. Xin is mainly in charge of finding more resources on github and the MPI implementation, while Shuyan is mainly in charge of searching for topics, ways to decompose a matrix, and CUDA implementation

### OpenMP
After testing the one-thread code on several sets of matrices, I started writting the parallel part of openMP. I tested the multi-threading on a 500*300 matrix, and tested the case of 1, 2,and 4 threads. Finally, the time it takes is shown in SVD_omp.out. Finally, for the multiple thread part, the time consumed has decreased. However, one should notice that even for the case of 1 single thread, the time consumed is already very low(around 2 seconds), so the improvement in efficiency doesn't seem to be that noticeable. As a result, I suppose if we want to further see the improvement in efficiency, it may be better if we can test it on a even larger matrix.

### MPI
MPI Scaling Results and Analysis
We ran the MPI-enabled Jacobi-based SVD on a computing cluster managed by Carc:

#### Strong Scaling (Fixed Problem Size)
Problem Setup: Matrix dimension fixed at 500 x 500, running 6 iterations.

![image](https://github.com/syzzzhou/CSCI-596-Project-parallel-computation-of-SVD/blob/Xin-He/images/Screenshot%202024-12-13%20174446.jpg)

Analysis: As we increase the number of processes from 1 to 4 for the same problem size, the runtime decreases significantly: From ~12.39 s with 1 process down to ~5.70 s with 4 processes. This shows good strong scaling: doubling the processes reduces computation time considerably. Data distribution overhead remains small, indicating efficient initialization and data partitioning.

#### Weak Scaling (Isogranular Scaling)
Problem Setup: Increase the matrix size proportionally to the number of processes, maintaining a similar workload per process. Six iterations were run for each configuration.
![image](https://github.com/syzzzhou/CSCI-596-Project-parallel-computation-of-SVD/blob/Xin-He/images/Screenshot%202024-12-13%20174513.jpg)

Analysis: As we increase both problem size and process count, total runtime increases from ~12 s (1 process) to ~20.6 s (4 processes). When performing weak scaling, the problem size and the number of processes both grow, keeping the per-process workload roughly constant. This is not we expected.

#### The Problem of Using Parallel Computing Matrix Decomposition:
1. More global communication: Operations like MPI_Allreduce occur more frequently and involve more processes, increasing communication overhead.
2. Longer iterations: Even if each process does roughly the same amount of work, the larger global problem size means more data to access, more rotations to compute, and extended synchronization periods, all contributing to a higher total runtime.

### CUDA
- **`cuda/`**  
  Contains the `jacobi.cu` file, which implements the Jacobi algorithm for Singular Value Decomposition (SVD) using CUDA for parallel computation.

- **Modified CPU-Based Implementation**  
  The `SVD_jacob.c` file has been updated to:
  - Accept matrix input from an external file for convenient testing with large matrices.
  - Generate and save the resulting decomposed matrices (`A`, `S`, `U`, `V`) into separate files for verification and further analysis.






