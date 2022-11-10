#!/bin/bash
#SBATCH --nodes=5                # node count
#SBATCH --ntasks=50              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G        # memory per cpu-core (4G is default)
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end,fail     # send email on job start, end and fault
#SBATCH --mail-user=mg4231@princeton.edu

module purge
module load julia/1.6.1

julia analyze_estimation.jl