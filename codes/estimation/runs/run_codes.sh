#!/bin/bash

file=estimation_fix_sigma0
rm -f jld/$file.txt

sbatch  $file.slurm
