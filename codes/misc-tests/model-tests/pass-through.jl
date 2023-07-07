β = 0.99^(1/3)
s = 0.031 
ψ = 1 - β*(1-s)

#pass-through
p1 = plot(x -> ψ*0.99^(1+1/x), 0.3, 3   , label="a = 0.99", xlabel=L"\varepsilon", ylabel = "pt")
plot!(p1, x -> ψ*1.01^(1+1/x), 0.3, 3, label = "a = 1.01"   )

# derivative 
p2 = plot(x -> ψ*(1+1/x)*0.99^(1/x), 0.3, 3, label = "a = 0.99", xlabel = L"\varepsilon", ylabel = "d pt / da")
plot!(p2, x -> ψ*(1+1/x)*1.01^(1/x), 0.3, 3, label = "a = 1.01")

plot(p1, p2,layout = (1,2),  size = (1000,400))