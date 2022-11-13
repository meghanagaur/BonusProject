using DelimitedFiles

cd(dirname(@__FILE__))

## Logistics
file_str       = "fix_hbar1"
file_load      = "jld/pretesting_"*file_str*".jld2"  # file to-load location
file_save      = "jld/estimation_"*file_str*".txt"   # file to-save 
N_procs        = 20                                  # number of jobs in job array

# Task number for job array
idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

println("JLD FILE = ", file_str)
include("run_smm.jl")