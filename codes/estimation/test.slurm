#!/bin/bash
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=20              # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

module purge
module load julia/1.6.1

julia -p  $SLURM_NTASKS -t  $SLURM_CPUS_PER_TASK  test.jl





