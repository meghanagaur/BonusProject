#!/bin/bash
#SBATCH --job-name=fix_a         # create a short name for your job
#SBATCH --output=slurm-%A.out    # stdout file
#SBATCH --error=slurm-%A.err     # stderr file
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=40              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=5:00:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end,fail     # send email when job ends
#SBATCH --mail-user=mg4231@princeton.edu

module purge
module load julia/1.6.1

julia pretesting_fix_a.jl 1.0