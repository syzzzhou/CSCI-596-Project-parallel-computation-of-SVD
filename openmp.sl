#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:00:59
#SBATCH --output=SVD_omp_iso.out
export OMP_NUM_THREADS=1
./SVD_omp 1
export OMP_NUM_THREADS=2
./SVD_omp 2
export OMP_NUM_THREADS=4
./SVD_omp 4
