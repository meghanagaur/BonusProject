#!/bin/bash

files="fix_hbar10_pt004 fix_hbar10_pt006 fix_hbar10_pt008 fix_hbar10_pt01"
#files="fix_hbar10_cyc05 fix_hbar10_cyc15 fix_hbar10_cyc20 fix_hbar10_cyc25 fix_hbar10_cyc30"

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
    cp estimation.slurm run-files/estimation_$file.slurm

    # replace file name
    sed -i "s/filename/$file/" "run-files/estimation_$file.jl"
    sed -i "s/filename/$file/" "run-files/estimation_$file.slurm"

    # submit job
    cd run-files
    sbatch estimation_$file.slurm
    cd ..
done

