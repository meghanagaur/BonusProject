#!/bin/bash
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # number of tasks per nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=00:00:05          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mg4231@princeton.edu

module purge
module load julia/1.6.1

julia -p 4 --threads $SLURM_CPUS_PER_TASK spmd_cluster_test.jl
#julia spmd_cluster_test.jl