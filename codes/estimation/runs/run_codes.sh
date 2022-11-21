#!/bin/bash

file=estimation_fix_eps03
rm -f jld/$file.txt

sbatch  $file.slurm
