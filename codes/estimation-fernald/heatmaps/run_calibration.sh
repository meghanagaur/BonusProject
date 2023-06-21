#!/bin/bash

files="calibration_vary_sigma_hbar calibration_vary_eps_hbar calibration_vary_eps_chi"

for file in $files
do
    sbatch  $file.slurm
done
