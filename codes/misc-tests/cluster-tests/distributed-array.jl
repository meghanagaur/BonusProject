
using Distributed

addprocs(3)
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD

d     = dzeros((100,100), workers()[1:3], [1,3])
d_in  = DArray(I -> fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1])
d_out = ddata();

@everywhere function foo_spmd(d_in, d_out, n)
   
    pids     = sort(vec(procs(d_in)))
    pididx   = findfirst(isequal(myid()), pids)
    mylp     = d_in[:L]
    localsum = 0

    # Have each worker exchange data with its neighbors
    n_pididx = pididx+1 > length(pids) ? 1 : pididx+1
    p_pididx = pididx-1 < 1 ? length(pids) : pididx-1

    for i in 1:n
        sendto(pids[n_pididx], mylp[2])
        sendto(pids[p_pididx], mylp[1])

        mylp[2] = recvfrom(pids[p_pididx])
        mylp[1] = recvfrom(pids[n_pididx])

        #barrier(;pids=pids)
        localsum = localsum + mylp[1] + mylp[2]
    end

    # finally store the sum in d_out
    d_out[:L] = localsum
end

spmd(foo_spmd,d_in,d_out, 5; pids=workers()) # executes on all workers


println(d_in)
println(d_out)


# test function
@everywhere function foo_spmd(d_in, d_out, n)
   
    pids     = sort(vec(procs(d_in)))
    pididx   = findfirst(isequal(myid()), pids)
    mylp     = d_in[:L]


    for i in 1:n
        mylp .+= ceil(Int64,rand()*10)
        d_out[:L] = d_in[1,1] 
    end

    # finally store the sum in d_out
 # Threads.nthreads() # maximum(d_in)
end
spmd(foo_spmd,d_in,d_out, 5; pids=workers()) # executes on all workers

#scp -r parallel-tests mg4231@adroit.princeton.edu:/scratch/network/mg4231
