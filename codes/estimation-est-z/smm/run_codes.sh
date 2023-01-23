#!/bin/bash

files="fix_rho_eps03"
#"fix_eps03_dlogw1_du_05 fix_eps03_dlogw1_du_1 fix_eps03_dlogw1_du_15 fix_eps03_dlogw1_du_2 fix_chi0"

for file in $files
do
    # remove and create output files
    rm -f jld/estimation_$file.txt
    touch jld/estimation_$file.txt

    # remove and create code files
    rm -f run-files/estimation_$file.jl
    cp estimation_v2.jl run-files/estimation_$file.jl

    # remove and create slurm files
    rm -f run-files/estimation_$file.slurm
    cp estimation.slurm run-files/estimation_$file.slurm

    # replace file name
    sed -i "s/filename/$file/" "run-files/estimation_$file.jl"
    sed -i "s/filename/$file/" "run-files/estimation_$file.slurm"

    # submit job
    cd run-files
    sbatch estimation_$file.slurm
    cd ..
done

