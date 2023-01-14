
using  DataFrames, CSV, DelimitedFiles

#file = "myfile.txt"
#writedlm(file,"f",",") #note: last argument is the delimiter which should be the same as below
#io = open(file, "a+", lock=true);

#=
for i = 1:10
    #sleep(5*rand())
    io = open(file, "a+", lock=true);
    write(io, string(i)*"\n");
    close(io);
    #println(i)
   
 #  CSV.write(file, DataFrame(f=i), header = false, append = true,delim=',')
end
=#

for i = 1:10

    sleep(5*rand())

    x = [1; 2; 3; 4];
    y = [1.1; 2.2; 3.3; 4.4];

    open("delim_file.txt", "a+") do io
            writedlm(io, [x y], ',')
        end;

    println(size(readdlm("delim_file.txt", ',', Float64),1))
    
end

out=vcat(fval, arg_min)

open(file, "a+") do io
    writedlm(io, out', ',')
end;

