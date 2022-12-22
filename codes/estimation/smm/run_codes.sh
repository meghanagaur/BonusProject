#!/bin/bash

files="dlogw1_du_05 dlogw1_du_1 dlogw1_du_15 dlogw1_du_2"

for file in $files
do
    # remove output files
    rm -f jld/estimation_$file.txt

    # remove and create code files
    rm -f run-files/estimation_$file.jl
    cp estimation.jl run-files/estimation_$file.jl

    # remove and create slurm files
    rm -f run-files/estimation_$file.slurm
    cp estimation.slurm run-files/estimation_$file.slurm

    # replace file name
    gsed -i "s/filename/$file/" "run-files/estimation_$file.jl"
    gsed -i "s/filename/$file/" "run-files/estimation_$file.slurm"

    # submit job
    sbatch  run-files/estimation_$file.slurm
done

