#!/bin/bash
#SBATCH --job-name=filename      # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=15       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=1-20             # job array with index values 0, 1, 2, 3, 4
#SBATCH --mail-type=end,fail     # send email on job start, end and fault
#SBATCH --mail-user=mg4231@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load julia/1.6.1

julia --threads $SLURM_CPUS_PER_TASK estimation_filename.jl






