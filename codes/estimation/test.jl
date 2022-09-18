

using Distributed, SlurmClusterManager
#addprocs(SlurmManager())
#num_tasks = parse(Int, ENV["SLURM_NTASKS"])
#addprocs(num_tasks)

println("num-workers",nworkers())

for i in workers()
   println("worker",i, "threads",@fetchfrom i Threads.nthreads() )
end















