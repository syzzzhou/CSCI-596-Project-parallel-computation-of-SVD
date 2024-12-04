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
