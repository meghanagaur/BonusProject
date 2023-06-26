# Produce preliminary heatmaps of calibration of σ_η and ε.
using LaTeXStrings, DataFrames, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8, titlefontsize =12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 12*Plots.px, legendfontsize = 9)

cd(dirname(@__FILE__))

# Load helper files
include("../functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.

# Get the parameter combination 
parameters   = collect(keys(get_param_bounds()))
combos       = collect(combinations(parameters, 2))

# Labeling of the parameters for plots
param_labels  = Dict{Symbol, LaTeXString}([   
    (:ε, L"\varepsilon"),
    (:σ_η, L"\sigma_{\eta}"),
    (:γ, L"\gamma"),
    (:χ, L"\chi") ]) 

# figure file path
dir  = "../figs/heatmaps/"
mkpath(dir)

# Loop through parameter combinations 
for combo in combos 

    # file path and parameters
    par1     = combo[1]
    par2     = combo[2]
    file     = string(par1)*"_"*string(par2)

    @unpack output, par_mat = load("jld/heatmap_vary_"*file*".jld2") 

    # Export data to make heatmaps
    N            = length(output)
    df           = DataFrame()
    df.par1      = par_mat[1,:] 
    df.par2      = par_mat[2,:]

    df.var_dlw       = [output[i][1][1] for i = 1:N]
    df.dlw1_du       = [output[i][1][2] for i = 1:N]
    df.dlw_dly       = [output[i][1][3] for i = 1:N]
    df.u_ss          = [output[i][1][4] for i = 1:N]
    df.std_u         = [output[i][1][5] for i = 1:N]
    df.dlogθ_dlogz   = [output[i][1][6] for i = 1:N]
    df.bwc_share     = [output[i][1][7] for i = 1:N]

    df.ir_flag       = [output[i][3] for i = 1:N]
    df.flag          = [output[i][2] for i = 1:N]
    df.ir_err        = [output[i][4] for i = 1:N]

    # Get back the parameter grids
    par1_grid    = unique(df.par1)
    par2_grid    = unique(df.par2)

    # Reshape moments into a conformable matrix
    var_dlw      = reshape(df.var_dlw, length(par1_grid), length(par2_grid) )
    dlw1_du      = reshape(df.dlw1_du, length(par1_grid), length(par2_grid) )
    dlw_dly      = reshape(df.dlw_dly, length(par1_grid), length(par2_grid) )
    u_ss         = reshape(df.u_ss,  length(par1_grid), length(par2_grid) )
    std_u        = reshape(df.std_u, length(par1_grid), length(par2_grid) )
    dlogθ_dlogz  = reshape(df.dlogθ_dlogz, length(par1_grid), length(par2_grid) )
    bwc_share    = reshape(df.bwc_share, length(par1_grid), length(par2_grid) )
    ir_flag      = reshape(df.ir_flag, length(par1_grid), length(par2_grid) )
    ir_err       = reshape(df.ir_err, length(par1_grid), length(par2_grid) )
    flag         = reshape(df.flag, length(par1_grid), length(par2_grid) )
    
    # Make plots
    par1_str = param_labels[par1]
    par2_str = param_labels[par2]

    # Plot var(dlw)
    p1 = heatmap(par1_grid, par2_grid, var_dlw, title="\n"*L"\textrm{Std. Dev.:} \Delta \log w_{t+12}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(p1, dir*"var_dlw_"*file*".pdf")

    # Plot dlw1/du
    p2 = heatmap(par1_grid, par2_grid, dlw1_du, title=L"\frac{ \partial E[ \log w_1 | z_t ]}{ \partial u_t}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(p2, dir*"dlw1_du.pdf")

    # Plot dlw/dly
    p3 = heatmap(par1_grid, par2_grid, dlw_dly,title=L"\mathbb{E}\left[\frac{\partial \log w_{it} }{ \partial \log y_{it} }\right]")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(p3, dir*"dlw_dly.pdf")

    # Plot \bar{u_t}
    p4 = heatmap(par1_grid, par2_grid, u_ss, title="\n"*L"\bar{u}_t")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(dir*"u_ss.pdf")

    # Plot BWC share at steady state
    std_u[ir_flag.==1] .= NaN
    p5 = heatmap(par1_grid, par2_grid, std_u, title="\n"*L"\textrm{Std. Dev.:} \log u_t")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(5, dir*"std_u.pdf")

    # Plot dlogθ/dlogz at steady state
    dlogθ_dlogz[ir_flag.==1] .= NaN
    p6 = heatmap(par1_grid, par2_grid, dlogθ_dlogz, title=L"\frac{d \log \theta }{ \partial \log z }")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(p6, dir*"dlogθ_dlogz.pdf")

    # Plot BWC share at steady state
    bwc_share[ir_flag.==1] .= NaN
    p7 = heatmap(par1_grid, par2_grid, bwc_share, title = "\n"*L"\textrm{BWC\ Share}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    savefig(p7, dir*"dlogθ_dlogz.pdf")

    # Plot IR_err
    p8 = heatmap(par1_grid, par2_grid, ir_err, title="\n"*L"\textrm{IR\ error}")
    xlabel!(par1_str)
    ylabel!(par2_str)

    # Plot IR flag
    p9 = heatmap(par1_grid, par2_grid, ir_flag, title="\n"*L"\textrm{IR\ flag}")
    xlabel!(par1_str)
    ylabel!(par2_str)

    plot(p8, p9, layout = (1,2),  size = (600,200))
    savefig(dir*"ir_error_flag.pdf")



   

