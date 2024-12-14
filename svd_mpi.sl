#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:59
#SBATCH --output=svd_mpi.out
#SBATCH -A anakano_429

echo "##### Strong scaling #####"
mpirun -n $SLURM_NTASKS ./svd_mpi
mpirun -n             2 ./svd_mpi
mpirun -n             1 ./svd_mpi

echo "##### Weak scaling   #####"
mpirun -n $SLURM_NTASKS ./svd_mpi_iso
mpirun -n             2 ./svd_mpi_iso
mpirun -n             1 ./svd_mpi_iso

