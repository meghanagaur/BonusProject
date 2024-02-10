"""
Taken from QuuantEcon:

Apply Hodrick-Prescott filter to `AbstractVector`.
##### Arguments
- `y::AbstractVector` : data to be detrended
- `λ::Real` : penalty on variation in trend
##### Returns
- `y_cyclical::Vector`: cyclical component
- `y_trend::Vector`: trend component
"""
function hp_filter(y::AbstractVector{T}, λ::Real) where T <: Real
    y = Vector(y)
    N = length(y)
    H = spdiagm(-2 => fill(λ, N-2),
                -1 => vcat(-2λ, fill(-4λ, N - 3), -2λ),
                 0 => vcat(1 + λ, 1 + 5λ, fill(1 + 6λ, N-4),
                           1 + 5λ, 1 + λ),
                 1 => vcat(-2λ, fill(-4λ, N - 3), -2λ),
                 2 => fill(λ, N-2))
    y_trend = float(H) \ y
    y_cyclical = y - y_trend
    return y_cyclical, y_trend
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y, X; intercept = true)
    
    if intercept == true
        XX = [ones(size(X,1)) X]
    else
        XX = X
    end    

    return  (XX'XX)\(XX'*Y)
    #return  inv(XX'XX)(XX'*YY)
end

"""
Approximate slope of y(x) by forward or central finite differences,
where y and x are both vectors.
"""
function slopeFD(y, x; diff = "central")
    if diff == "forward"
        return (y[2:end] - y[1:end-1])./(x[2:end] - x[1:end-1])
        dydx = [dydx ; NaN]
    elseif diff == "central" 
        dydx = (y[3:end] - y[1:end-2])./(x[3:end] - x[1:end-2])
        dydx = [NaN; dydx; NaN]
    elseif diff == "backward" 
        return (y[2:end] - y[1:end-1])./(x[2:end] - x[1:end-1])
        dydx = [NaN; dydx] 
    end
    return dydx
end

"""
Law of motion for unemployment
given u0, returns u1
"""
function uLM(u0, s, f)
    u1 = u0 + s*(1 - u0) - (1-s)*f*u0
    return u1
end

"""
Return quarterly average of monthly series x_t, length T_q
"""
function quarterlyAverage(x_t, T_q; weights = nothing)
    if weights == nothing
        return [mean(x_t[(t_q*3 - 2):t_q*3]) for t_q = 1:T_q] 
    else
        total1 = [sum(weights[(t_q*3 - 2):t_q*3]) for t_q = 1:T_q]
        total2 = [sum(x_t[(t_q*3 - 2):t_q*3].*weights[(t_q*3 - 2):t_q*3]) for t_q = 1:T_q]
        return total1./total2 
    end
end