#!/bin/bash

files="fix_a_bwc10_est_z fix_a_bwc10_pv_est_z fix_a_bwc10_fix_wages_est_z"

for file in $files
do
    # remove and create output files
    rm -f jld/estimation_$file.txt
    touch jld/estimation_$file.txt

    # remove and create code files
    rm -f run-files/estimation_$file.jl
    cp estimation.jl run-files/estimation_$file.jl

    # remove and create slurm files
    rm -f run-files/estimation_$file.slurm
    cp estimation_fix_a.slurm run-files/estimation_$file.slurm

    # replace file name
    sed -i "s/filename/$file/" "run-files/estimation_$file.jl"
    sed -i "s/filename/$file/" "run-files/estimation_$file.slurm"

    # submit job
    cd run-files
    sbatch estimation_$file.slurm
    cd ..
done
