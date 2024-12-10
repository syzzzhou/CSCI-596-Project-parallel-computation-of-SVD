#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 1 process per computing node
#SBATCH --cpus-per-task=2 2 cores (threads) per process
#SBATCH --time=00:00:59
#SBATCH --output=omp_example.out
export OMP_NUM_THREADS=2