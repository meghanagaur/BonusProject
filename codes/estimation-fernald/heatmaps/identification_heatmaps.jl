# Produce preliminary heatmaps of calibration of σ_η and ε.
using LaTeXStrings, DataFrames, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =15, yguidefontsize=15, xtickfontsize=10, ytickfontsize=10, titlefontsize =12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 15*Plots.px, legendfontsize = 9)

cd(dirname(@__FILE__))

# Load helper files
include("../functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.

# Get the parameter combinations
parameters   = collect(keys(get_param_bounds()))
combos       = collect(combinations(parameters, 2))

# Labeling of the parameters for plots
param_labels  = Dict{Symbol, LaTeXString}([   
    (:ε, L"\epsilon"),
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

    @unpack output, par_mat, baseline_params = load("jld/heatmap_vary_"*file*".jld2") 

    # Import data to make heatmaps
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

    df.flag          = [output[i][2] for i = 1:N]
    df.ir_flag       = [output[i][3] for i = 1:N]
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
    var_dlw[ir_flag.==1] .= NaN
    p1 = heatmap(par1_grid, par2_grid, var_dlw) #, title="\n"*L"\textrm{Std. Dev.:} \Delta \log w_{t+12}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p1, dir*"std_dlw_"*file*".pdf")

    # Plot dlw1/du
    dlw1_du[ir_flag.==1] .= NaN
    p2 = heatmap(par1_grid, par2_grid, dlw1_du) #, title=L"\frac{ \partial E[ \log w_1 | z_t ]}{ \partial u_t}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p2, dir*"dlw1_du_"*file*".pdf")

    # Plot dlw/dly
    dlw_dly[ir_flag.==1] .= NaN
    p3 = heatmap(par1_grid, par2_grid, dlw_dly) #,title=L"\mathbb{E}\left[\frac{\partial \log w_{it} }{ \partial \log y_{it} }\right]")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p3, dir*"dlw_dly_"*file*".pdf")

    # Plot \bar{u_t}
    u_ss[ir_flag.==1] .= NaN
    p4 = heatmap(par1_grid, par2_grid, u_ss) #, title="\n"*L"\bar{u}_t")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(dir*"u_ss_"*file*".pdf")

    # Plot BWC share at steady state
    std_u[ir_flag.==1] .= NaN
    p5 = heatmap(par1_grid, par2_grid, std_u) #, title="\n"*L"\textrm{Std. Dev.:} \log u_t")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p5, dir*"std_u_"*file*".pdf")

    # Plot dlogθ/dlogz at steady state
    dlogθ_dlogz[ir_flag.==1] .= NaN
    p6 = heatmap(par1_grid, par2_grid, dlogθ_dlogz) #, title=L"\frac{d \log \theta }{ \partial \log z }")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p6, dir*"dlogtheta_dlogz_"*file*".pdf")

    # Plot BWC share at steady state
    bwc_share[ir_flag.==1] .= NaN
    p7 = heatmap(par1_grid, par2_grid, bwc_share) #, title = "\n"*L"\textrm{BWC\ Share}")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)
    savefig(p7, dir*"bwc_share_"*file*".pdf")

    # Plot IR_err
    p8 = heatmap(par1_grid, par2_grid, ir_err) #, title="\n"*L"\textrm{TIOLI/PC\ Error}"*"\n")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)

    # Plot IR flag
    p9 = heatmap(par1_grid, par2_grid, ir_flag) #, title="\n"*L"\textrm{TIOLI/PC\ Flag}"*"\n")
    xlabel!(par1_str)
    ylabel!(par2_str)
    annotate!([baseline_params[par1]], [baseline_params[par2]], "X", annotationcolor=:green)

    plot(p8, p9, layout = (1,2),  size = (1000,400))
    savefig(dir*"ir_error_flag_"*file*".pdf")
end

   

