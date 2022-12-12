#!/bin/bash

files="estimation_fix_eps02 estimation_fix_eps03 estimation_fix_eps04 estimation_fix_eps05 pretesting_fix_hbar1"

for file in $files
do
    rm -f jld/$file.txt
    sbatch  $file.slurm
done
