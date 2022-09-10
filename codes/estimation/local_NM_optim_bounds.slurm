#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=11       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mg4231@princeton.edu

module purge
module load julia/1.6.1

julia --threads $SLURM_CPUS_PER_TASK local_NM_optim_bounds.jl