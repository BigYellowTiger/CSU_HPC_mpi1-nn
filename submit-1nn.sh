#!/bin/bash

#SBATCH --job-name=poker100c
#SBATCH -o poker100c.out
#SBATCH -e poker100c.err
#SBATCH -N 10
#SBATCH --ntasks-per-node=10

echo -e '\n submitted Open MPI job'
echo 'hostname'
hostname

module load mvapich2-2.3.3
PY3=/public/software/Python-3.9.2/bin/python3
mpiexec $PY3 -u 1-nn-mpi.py
