using Distributed, SharedArrays, DistributedArrays, CSV,  DelimitedFiles
cd("/Users/meghanagaur/BonusProject/codes/parallel-tests")

addprocs(3)

# practice using distributed loop
@everywhere g(x) = x

a = SharedArray{Float64}(10)
@distributed for i = 1:10
    a[i] = g(i)
end

# practice using pmap
@everywhere function g!(x,a)
    if x>1
        if x >2
        sleep(rand()*5)
        end
        a[x] = a[x-1]
    else
        a[x]=x
    end
end

a = SharedArray{Float64}(10)
pmap(x->g!(x,a), 1:length(a))


# try file i-o

@everywhere using  DataFrames, CSV, DelimitedFiles

@everywhere file = "myfile.txt"
writedlm(file,"f",",") #note: last argument is the delimiter which should be the same as below


 @time @sync @distributed for i = 1:10
    sleep(5*rand())
    io = open(file, "a+", lock=true);
    write(io, string(i)*"\n");
    close(io);
    #println(i)
   
 #  CSV.write(file, DataFrame(f=i), header = false, append = true,delim=',')
end






