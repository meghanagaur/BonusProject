#!/bin/bash

#files="baseline fix_chi0" 
#files="cyc05 cyc075 cyc125 cyc15"
#files="fix_eps2.4"
#files="fix_chi0"
files=" fix_eps05 fix_eps10 fix_eps15 fix_eps20 fix_eps25 fix_eps30 fix_eps35 fix_eps40 fix_eps45 fix_eps50"

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

