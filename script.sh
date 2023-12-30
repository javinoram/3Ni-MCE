#!/bin/bash
#SBATCH --job-name=3DMolISO      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=javier.norambuenal@sansano.usm.cl    # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run two tasks		
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=8gb                    # Job memory request
#SBATCH --time=200:00:00              # Time limit hrs:min:sec
#SBATCH --output=output/parallel_%j.log     # Standard output and error log
#SBATCH --partition=beta

source /opt/intel/oneapi/setvars.sh

# Primera tarea: ejecutar el primer archivo
mpirun -n 1 python3 two-mol/ISOQuen.py 1
