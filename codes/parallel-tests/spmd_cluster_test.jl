using Distributed, SlurmClusterManager

println(nworkers())
println(Threads.nthreads())

#num_tasks = parse(Int, ENV["SLURM_NTASKS_PER_NODE"])
#num_nodes = parse(Int, ENV["SLURM_NODES"])
#addprocs(num_tasks*num_nodes)
#addprocs(SlurmManager())
#addprocs(3)

@everywhere println("hello from $(myid()):$(gethostname())")
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD

d_in  = DArray(I -> fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1])
#d_out = DArray(I -> fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1])
d_out  = ddata();

# threaded function
@everywhere function foo_spmd_thread(d_in, d_out, n)
   
    pids     = sort(vec(procs(d_in)))
    pididx   = findfirst(isequal(myid()), pids)
    localsum = d_in[:L]

    Threads.@threads for i = 1:n
        sleep(10)
    end
    # finally store the sum in d_out
     d_out[:L] =  Threads.nthreads()
end


@time spmd(foo_spmd_thread, d_in,d_out, 5; pids=workers()) # executes on all workers
println(d_in)
println(d_out)

# test function
@everywhere function foo_spmd(d_in, d_out, n)
   
    pids     = sort(vec(procs(d_in)))
    pididx   = findfirst(isequal(myid()), pids)
    localsum = d_in[:L]
    localsum  = 3
    for i = 1:n
       sleep(10)
    end
    # finally store the sum in d_out
     d_out[:L] = Threads.nthreads()
end


@time spmd(foo_spmd, d_in,d_out, 5; pids=workers()) # executes on all workers
println(d_in)
println(d_out)

